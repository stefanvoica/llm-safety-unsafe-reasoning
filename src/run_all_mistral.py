import os, sys, re, json, argparse, datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, AutoTokenizer, AutoModelForCausalLM,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURARE
# ─────────────────────────────────────────────────────────────────────────────
BASE        = os.path.expanduser("~/project/data")
MODEL_DIR   = os.path.expanduser("~/project/model")
LOG_DIR     = os.path.expanduser("~/project/logs")
MODEL_NAME  = "mistralai/Mistral-3B-Instruct-v0.3"
MAX_LENGTH  = 1024
BATCH_SIZE  = 4      
GRAD_ACCUM  = 2     
EPOCHS      = 3
ALL_RANKS   = [4, 16, 32, 64]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,   exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "w", buffering=1)
        self.log.write(f"=== Start: {datetime.datetime.now()} ===\n\n")

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.write(f"\n=== End: {datetime.datetime.now()} ===\n")
        self.log.close()

def set_log(name):
    """Redirectează stdout spre fișierul de log și îl returnează."""
    path = os.path.join(LOG_DIR, name)
    logger = Logger(path)
    sys.stdout = logger
    print(f"Logging în: {path}")
    return logger

def reset_log(logger):
    logger.close()
    sys.stdout = logger.terminal

def load_data():
    df_unsafe     = pd.read_json(f"{BASE}/Train/train_unsafe.jsonl",            lines=True)
    df_safe       = pd.read_json(f"{BASE}/Train/train_safe.jsonl",              lines=True)
    df_potentially= pd.read_json(f"{BASE}/Train/train_potentially_unsafe.jsonl",lines=True)
    df_full       = pd.concat([df_unsafe, df_safe, df_potentially], ignore_index=True)

    train_df, val_df = train_test_split(
        df_full, test_size=0.10, random_state=42,
        stratify=df_full['label'] if 'label' in df_full.columns else None
    )

    dv_unsafe     = pd.read_json(f"{BASE}/Validation/valid_unsafe.jsonl",            lines=True)
    dv_safe       = pd.read_json(f"{BASE}/Validation/valid_safe.jsonl",              lines=True)
    dv_potentially= pd.read_json(f"{BASE}/Validation/valid_potentially_unsafe.jsonl",lines=True)
    test_df       = pd.concat([dv_unsafe, dv_safe, dv_potentially], ignore_index=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"safe": 0, "potentially unsafe": 1, "unsafe": 2}

class SafetyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length= max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        query = str(row.get('query', ''))
        trace = str(row.get('reasoning_trace', ''))
        text  = f"Query: {query}\nReasoning Trace:\n{trace}"

        enc = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        label_str = str(row.get('label', 'safe')).strip().lower()
        label_idx = LABEL_MAP.get(label_str, 0)

        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(label_idx, dtype=torch.long)
        }

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class SafetyClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=3, r_value=8):
        super().__init__()
        # float16 + low_cpu_mem_usage: Mistral-3B ~6GB pe disk,
        # fără float16 nu încape pe GPU-uri de 16GB împreună cu gradienți
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        lora_cfg = LoraConfig(
            r=r_value, lora_alpha=2*r_value,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.1, bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.base_model = get_peft_model(self.base_model, lora_cfg)
        self.base_model.print_trainable_parameters()
        hidden = self.base_model.config.hidden_size  # 3072 pentru Mistral-3B
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out  = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        lhs  = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
        # mean pooling — media hidden states ale tuturor token-urilor non-padding
        vec  = torch.sum(lhs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.classifier(vec.float())  # cast la float32 pt classifier

# ─────────────────────────────────────────────────────────────────────────────
# ANTRENARE
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_ds, val_ds, epochs, save_path):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
    model        = model.to(device)

    counts  = Counter(train_ds.data['label'].str.strip().str.lower())
    total   = sum(counts.values())
    weights = torch.tensor([
        total / max(counts.get('safe', 1), 1),
        total / max(counts.get('potentially unsafe', 1), 1),
        total / max(counts.get('unsafe', 1), 1),
    ], dtype=torch.float).to(device)
    print(f"Class weights: safe={weights[0]:.2f}, pot_unsafe={weights[1]:.2f}, unsafe={weights[2]:.2f}")

    criterion = CrossEntropyLoss(weight=weights)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)
    # total_steps calculat pe baza pașilor efectivi (după acumulare)
    total_steps = (len(train_loader) // GRAD_ACCUM) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n=== Epoca {epoch+1}/{epochs} ===")
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Train")):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)

            loss = criterion(model(ids, mask), lbls) / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item() * GRAD_ACCUM

            # optimizer.step() doar după GRAD_ACCUM pași
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Train loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss, correct, total_ex = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                logits = model(ids, mask)
                val_loss += criterion(logits, lbls).item()
                correct  += (torch.argmax(logits,-1) == lbls).sum().item()
                total_ex += lbls.size(0)
        avg_val = val_loss / len(val_loader)
        val_mf1 = f1_score(
            [b['labels'].item() for b in val_ds],
            [], average='macro', zero_division=0
        ) if False else 0.0  # placeholder, calculat mai jos
        print(f"Val loss: {avg_val:.4f} | Val acc: {correct/total_ex*100:.2f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"✅ Salvat best model (val_loss={avg_val:.4f})")

    print(f"\n🏁 Done. Best val_loss: {best_val_loss:.4f}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# EVALUARE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, test_ds, r_value=8):
    model = SafetyClassifier(r_value=r_value).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluare"):
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            preds.extend(torch.argmax(logits,-1).cpu().numpy())
            labels.extend(batch['labels'].numpy())

    acc    = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\nAccuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds,
          labels=[0,1,2], target_names=['safe','potentially_unsafe','unsafe'],
          zero_division=0))
    return acc, macro_f1

# ─────────────────────────────────────────────────────────────────────────────
# Q1 — ZERO-SHOT
# ─────────────────────────────────────────────────────────────────────────────
def run_zeroshot(test_df):
    log = set_log("q1_zeroshot.txt")
    print("=== Q1 — Zero-Shot Mistral-3B-Instruct ===\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    zs_model  = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    zs_model.eval()

    def prompt(query, traj):
        return (f"Read the following interaction:\nQuery: {query}\n"
                f"Reasoning Trajectory: {traj}\n\n"
                "Classify this interaction as one of:\n"
                "- safe: content is completely harmless.\n"
                "- potentially unsafe: content touches sensitive topics.\n"
                "- unsafe: content provides directly dangerous instructions.\n\n"
                "Respond ONLY with the class name (safe, potentially unsafe or unsafe).\nClass: ")

    correct = 0
    all_preds, all_labels = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        query = str(row.get('query',''))
        traj  = str(row.get('reasoning_trace',''))
        p     = prompt(query, traj)
        inp   = tokenizer(p, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            out = zs_model.generate(**inp, max_new_tokens=10, temperature=0.1,
                                     pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inp['input_ids'].shape[1]:],
                               skip_special_tokens=True).strip().lower()

        if "potentially unsafe" in gen:
            pred = 1
        elif "unsafe" in gen:
            pred = 2
        else:
            pred = 0

        true = LABEL_MAP.get(str(row.get('label','safe')).strip().lower(), 0)
        all_preds.append(pred); all_labels.append(true)
        if pred == true: correct += 1

    acc = correct / len(test_df)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    print(f"\nAccuracy: {acc*100:.2f}% | Macro-F1: {mf1:.4f}")
    print(classification_report(all_labels, all_preds,
          labels=[0,1,2], target_names=['safe','potentially_unsafe','unsafe'],
          zero_division=0))
    reset_log(log)

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — BUCKETS DE LUNGIME
# ─────────────────────────────────────────────────────────────────────────────
def count_steps(trace):
    return len(re.findall(r'step\s+\d+', str(trace), re.IGNORECASE))

def run_q4(model_path, test_df, tokenizer, r_value=8):
    log = set_log("q4_buckets.txt")
    print("=== Q4 — Performanță pe Buckets de Lungime ===\n")

    model = SafetyClassifier(r_value=r_value).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_df = test_df.copy()
    test_df['n_steps'] = test_df['reasoning_trace'].apply(count_steps)

    def bucket(n):
        if n <= 5:  return "Short (1-5)"
        if n <= 10: return "Medium (6-10)"
        if n <= 15: return "Long (11-15)"
        return "Very Long (15+)"

    test_df['bucket'] = test_df['n_steps'].apply(bucket)

    for b in ["Short (1-5)", "Medium (6-10)", "Long (11-15)", "Very Long (15+)"]:
        sub = test_df[test_df['bucket'] == b]
        if len(sub) == 0:
            print(f"{b}: 0 exemple\n"); continue
        ds = SafetyDataset(sub, tokenizer)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                preds.extend(torch.argmax(logits,-1).cpu().numpy())
                labels.extend(batch['labels'].numpy())
        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average='macro', zero_division=0)
        print(f"{b} | n={len(sub):4d} | Acc={acc*100:.2f}% | Macro-F1={mf1:.4f}")

    reset_log(log)

# ─────────────────────────────────────────────────────────────────────────────
# Q3 — INTEGRATED GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────
def run_q3(model_path, test_df, tokenizer, r_value=8):
    log = set_log("q3_ig.txt")
    print("=== Q3 — Integrated Gradients ===\n")

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'captum', '-q'])
        from captum.attr import IntegratedGradients

    model = SafetyClassifier(r_value=r_value).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    class CaptumWrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, embeds, mask):
            out  = self.m.base_model(inputs_embeds=embeds, attention_mask=mask)
            lhs  = out.last_hidden_state
            msk  = mask.unsqueeze(-1).expand(lhs.size()).float()
            vec  = torch.sum(lhs*msk,1) / torch.clamp(msk.sum(1), min=1e-9)
            return self.m.classifier(vec)

    def explain(text, target_class=0):
        model.eval()
        wrapper = CaptumWrapper(model).to(device)
        inp  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        ids  = inp['input_ids'].to(device)
        mask = inp['attention_mask'].to(device)
        emb_layer = model.base_model.get_input_embeddings()
        embeds   = emb_layer(ids)
        baseline = emb_layer(torch.full_like(ids, tokenizer.pad_token_id))
        ig = IntegratedGradients(wrapper)
        attrs, _ = ig.attribute(embeds, baseline, additional_forward_args=(mask,),
                                target=target_class, n_steps=50, internal_batch_size=2,
                                return_convergence_delta=True)
        attrs_sum = attrs.sum(-1).squeeze(0)
        attrs_sum = attrs_sum / torch.norm(attrs_sum)
        tokens = tokenizer.convert_ids_to_tokens(ids[0])
        return [(t.replace('Ġ',' '), s.item()) for t,s in zip(tokens, attrs_sum.cpu().detach())]

    def aggregate(word_attrs):
        agg = {"Query": 0.0}; cur = "Query"
        for tok, score in word_attrs:
            if "step" in tok.lower():
                nums = re.findall(r'\d+', tok.lower())
                cur  = f"Step {nums[0]}" if nums else "Step X"
                if cur not in agg: agg[cur] = 0.0
            if cur in agg: agg[cur] += score
        return agg

    # Găsim TP, TN, FP, FN
    found = {"TP": None, "TN": None, "FP": None, "FN": None}
    for _, row in test_df.iterrows():
        if all(v is not None for v in found.values()): break
        text = f"Query: {row.get('query','')}\nReasoning Trace:\n{row.get('reasoning_trace','')}"
        true = str(row.get('label','safe')).strip().lower()
        inp  = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            pred_idx = torch.argmax(model(inp['input_ids'], inp['attention_mask']), -1).item()
        pred = ["safe","potentially unsafe","unsafe"][pred_idx]
        if true=="safe" and pred=="safe"       and found["TP"] is None: found["TP"]=text
        elif true!="safe" and pred!="safe"     and found["TN"] is None: found["TN"]=text
        elif true!="safe" and pred=="safe"     and found["FP"] is None: found["FP"]=text
        elif true=="safe" and pred!="safe"     and found["FN"] is None: found["FN"]=text

    for cat, text in found.items():
        if text is None: print(f"\n{cat}: niciun exemplu găsit."); continue
        print(f"\n{'='*50}\nCategorie: {cat}\n{'='*50}")
        attrs = explain(text, target_class=0)
        agg   = aggregate(attrs)
        for sec, score in agg.items():
            label = "→ SAFE (+)" if score > 0 else "→ UNSAFE (-)"
            print(f"{sec.ljust(10)} | {score:>8.4f} | {label}")

    reset_log(log)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks",         nargs="+", type=int, default=ALL_RANKS)
    parser.add_argument("--epochs",        type=int,  default=EPOCHS)
    parser.add_argument("--skip-zeroshot", action="store_true")
    parser.add_argument("--skip-q3",       action="store_true")
    parser.add_argument("--skip-q4",       action="store_true")
    parser.add_argument("--eval-only",     action="store_true",
                        help="Sare antrenarea, doar evaluează modelele existente")
    args = parser.parse_args()

    print(f"Device: {device}\n")
    train_df, val_df, test_df = load_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    val_ds  = SafetyDataset(val_df,   tokenizer)
    test_ds = SafetyDataset(test_df,  tokenizer)

    # ── Q1 + Q2: antrenare pe fiecare rank ───────────────────────────────────
    results = {}
    for r in args.ranks:
        model_path = os.path.join(MODEL_DIR, f"mistral_safety_model_r{r}.pt")
        log_name   = "q1_finetuned.txt" if r == min(args.ranks) else f"q2_r{r}.txt"

        log = set_log(log_name)
        print(f"=== Rank r={r} ===\n")

        if not args.eval_only:
            train_ds = SafetyDataset(train_df, tokenizer)
            model    = SafetyClassifier(r_value=r)
            train_model(model, train_ds, val_ds, args.epochs, model_path)
            del model; torch.cuda.empty_cache()

        if os.path.exists(model_path):
            print(f"\n--- Evaluare test set (r={r}) ---")
            acc, mf1 = evaluate_model(model_path, test_ds, r_value=r)
            results[r] = (acc, mf1)
        else:
            print(f"Model r={r} nu există la {model_path}, skip evaluare.")

        reset_log(log)

    # Sumarul Q2 — în consolă și în fișier text
    if results:
        log = set_log("q2_summary.txt")
        print("=== Q2 — Sumar Macro-F1 per Rank ===\n")
        print(f"{'Rank':>6} | {'Accuracy':>9} | {'Macro-F1':>9}")
        print("-" * 32)
        for r, (acc, mf1) in sorted(results.items()):
            print(f"r={r:>4} | {acc*100:>8.2f}% | {mf1:>9.4f}")
        reset_log(log)

        # ── Salvare JSON cu toate rezultatele ────────────────────────────────
        summary = {
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "epochs": args.epochs,
            "timestamp": str(datetime.datetime.now()),
            "results": {
                f"r{r}": {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}
                for r, (acc, mf1) in sorted(results.items())
            },
            "best_rank": max(results, key=lambda r: results[r][1])
        }
        summary_path = os.path.join(LOG_DIR, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary JSON salvat: {summary_path}")
        print(f"🏆 Best rank: r={summary['best_rank']} "
              f"(Macro-F1={results[summary['best_rank']][1]:.4f})")

    # ── Q1 Zero-Shot ─────────────────────────────────────────────────────────
    if not args.skip_zeroshot:
        run_zeroshot(test_df)

    # ── Q3 + Q4: folosește automat cel mai bun rank din rezultate ────────────
    if results:
        best_r = max(results, key=lambda r: results[r][1])
    else:
        best_r = args.ranks[0]
    best_path = os.path.join(MODEL_DIR, f"mistral_safety_model_r{best_r}.pt")
    print(f"\nUsing best model: r={best_r} pentru Q3/Q4")

    if not args.skip_q3 and os.path.exists(best_path):
        run_q3(best_path, test_df, tokenizer, r_value=best_r)

    if not args.skip_q4 and os.path.exists(best_path):
        run_q4(best_path, test_df, tokenizer, r_value=best_r)

    print("\n✅ Toate experimentele finalizate. Loguri în ~/project/logs/")

if __name__ == "__main__":
    main()
