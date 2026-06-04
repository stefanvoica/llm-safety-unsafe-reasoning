import os, sys, re, json, argparse, datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURARE
# ─────────────────────────────────────────────────────────────────────────────
BASE      = os.path.expanduser("~/project/data")
MODEL_DIR = os.path.expanduser("~/project/model")
LOG_DIR   = os.path.expanduser("~/project/logs")

CONFIGS = {
    "base": {
        "model_name": "bert-base-uncased",
        "batch_size": 32,
        "grad_accum": 1,    # batch efectiv = 32
        "lr":         2e-5,
        "dtype":      torch.float32,
    },
    "large": {
        "model_name": "bert-large-uncased",
        "batch_size": 16,
        "grad_accum": 2,    # batch efectiv = 32
        "lr":         1e-5,
        "dtype":      torch.float32,
    },
}

MAX_LENGTH = 512
EPOCHS     = 3

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

    def isatty(self):
        return False

    def close(self):
        self.log.write(f"\n=== End: {datetime.datetime.now()} ===\n")
        self.log.close()

def set_log(name):
    path   = os.path.join(LOG_DIR, name)
    logger = Logger(path)
    sys.stdout = logger
    print(f"Logging în: {path}")
    return logger

def reset_log(logger):
    logger.close()
    sys.stdout = logger.terminal

# ─────────────────────────────────────────────────────────────────────────────
# DATE
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    df_unsafe      = pd.read_json(f"{BASE}/Train/train_unsafe.jsonl",             lines=True)
    df_safe        = pd.read_json(f"{BASE}/Train/train_safe.jsonl",               lines=True)
    df_potentially = pd.read_json(f"{BASE}/Train/train_potentially_unsafe.jsonl", lines=True)
    df_full        = pd.concat([df_unsafe, df_safe, df_potentially], ignore_index=True)

    train_df, val_df = train_test_split(
        df_full, test_size=0.10, random_state=42,
        stratify=df_full['label'] if 'label' in df_full.columns else None
    )

    dv_unsafe      = pd.read_json(f"{BASE}/Validation/valid_unsafe.jsonl",             lines=True)
    dv_safe        = pd.read_json(f"{BASE}/Validation/valid_safe.jsonl",               lines=True)
    dv_potentially = pd.read_json(f"{BASE}/Validation/valid_potentially_unsafe.jsonl", lines=True)
    test_df        = pd.concat([dv_unsafe, dv_safe, dv_potentially], ignore_index=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df

# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {"safe": 0, "potentially unsafe": 1, "unsafe": 2}

class SafetyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.data       = dataframe.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        query = str(row.get('query', ''))
        trace = str(row.get('reasoning_trace', ''))
        # BERT suportă perechi de segmente → tokenizer adaugă [CLS] query [SEP] trace [SEP]
        enc = self.tokenizer(
            query, trace,
            truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt"
        )
        label_str = str(row.get('label', 'safe')).strip().lower()
        label_idx = LABEL_MAP.get(label_str, 0)

        item = {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'labels':         torch.tensor(label_idx, dtype=torch.long)
        }
        if 'token_type_ids' in enc:
            item['token_type_ids'] = enc['token_type_ids'].flatten()
        return item

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=3, dtype=torch.float32):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            ignore_mismatched_sizes=True
        )
        hidden = self.backbone.config.hidden_size
        print(f"hidden_size: {hidden}")
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs['token_type_ids'] = token_type_ids
        out = self.backbone(**kwargs)
        # [CLS] token — primul token, antrenat pentru clasificare
        cls_vec = out.last_hidden_state[:, 0, :]
        return self.classifier(cls_vec.float())

# ─────────────────────────────────────────────────────────────────────────────
# ANTRENARE
# ─────────────────────────────────────────────────────────────────────────────
def train_model(model, train_ds, val_ds, cfg, epochs, save_path):
    bs         = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    lr         = cfg["lr"]

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              num_workers=2, pin_memory=True)
    model        = model.to(device)

    counts  = Counter(train_ds.data['label'].str.strip().str.lower())
    total   = sum(counts.values())
    weights = torch.tensor([
        total / max(counts.get('safe', 1), 1),
        total / max(counts.get('potentially unsafe', 1), 1),
        total / max(counts.get('unsafe', 1), 1),
    ], dtype=torch.float).to(device)
    print(f"Class weights: safe={weights[0]:.2f}, pot_unsafe={weights[1]:.2f}, unsafe={weights[2]:.2f}")
    print(f"Batch size: {bs} | Grad accum: {grad_accum} | LR: {lr} | Batch efectiv: {bs*grad_accum}")

    criterion   = CrossEntropyLoss(weight=weights)
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // max(grad_accum, 1)) * epochs
    scheduler   = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n=== Epoca {epoch+1}/{epochs} ===")
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc="Train")):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['labels'].to(device)
            ttids = batch.get('token_type_ids')
            if ttids is not None:
                ttids = ttids.to(device)

            logits = model(ids, mask, ttids)
            loss   = criterion(logits, lbls) / max(grad_accum, 1)
            loss.backward()
            total_loss += loss.item() * max(grad_accum, 1)

            if grad_accum <= 1 or (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        print(f"Train loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        val_loss, correct, total_ex = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val"):
                ids   = batch['input_ids'].to(device)
                mask  = batch['attention_mask'].to(device)
                lbls  = batch['labels'].to(device)
                ttids = batch.get('token_type_ids')
                if ttids is not None:
                    ttids = ttids.to(device)
                logits    = model(ids, mask, ttids)
                val_loss += criterion(logits, lbls).item()
                correct  += (torch.argmax(logits, -1) == lbls).sum().item()
                total_ex += lbls.size(0)

        avg_val = val_loss / len(val_loader)
        print(f"Val loss: {avg_val:.4f} | Val acc: {correct/total_ex*100:.2f}%")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            print(f"✅ Salvat best model (val_loss={avg_val:.4f})")

    print(f"\n🏁 Done. Best val_loss: {best_val_loss:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# EVALUARE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(model_path, test_ds, model_name, dtype, bs=16):
    model = BERTClassifier(model_name=model_name, dtype=dtype).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluare"):
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttids = batch.get('token_type_ids')
            if ttids is not None:
                ttids = ttids.to(device)
            logits = model(ids, mask, ttids)
            preds.extend(torch.argmax(logits, -1).cpu().numpy())
            labels.extend(batch['labels'].numpy())

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\nAccuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds,
          labels=[0, 1, 2], target_names=['safe', 'potentially_unsafe', 'unsafe'],
          zero_division=0))
    return acc, macro_f1

# ─────────────────────────────────────────────────────────────────────────────
# MAJORITY BASELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_majority_baseline(test_df):
    log = set_log("bert_majority_baseline.txt")
    print("=== Majority Class Baseline ===\n")
    labels   = [LABEL_MAP.get(str(r).strip().lower(), 0) for r in test_df['label']]
    counts   = Counter(labels)
    majority = counts.most_common(1)[0][0]
    preds    = [majority] * len(labels)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"Majority class: {['safe','potentially unsafe','unsafe'][majority]}")
    print(f"Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds,
          labels=[0, 1, 2], target_names=['safe', 'potentially_unsafe', 'unsafe'],
          zero_division=0))
    reset_log(log)

# ─────────────────────────────────────────────────────────────────────────────
# Q4 — BUCKETS DE LUNGIME
# ─────────────────────────────────────────────────────────────────────────────
def count_steps(trace):
    return len(re.findall(r'step\s+\d+', str(trace), re.IGNORECASE))

def run_q4(model_path, test_df, tokenizer, model_name, dtype, variant, bs=16):
    log = set_log(f"bert_{variant}_q4_buckets.txt")
    print(f"=== Q4 — Buckets de Lungime ({model_name}) ===\n")

    model = BERTClassifier(model_name=model_name, dtype=dtype).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_df            = test_df.copy()
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
        ds     = SafetyDataset(sub, tokenizer)
        loader = DataLoader(ds, batch_size=bs, shuffle=False)
        preds, labels = [], []
        with torch.no_grad():
            for batch in loader:
                ids   = batch['input_ids'].to(device)
                mask  = batch['attention_mask'].to(device)
                ttids = batch.get('token_type_ids')
                if ttids is not None:
                    ttids = ttids.to(device)
                logits = model(ids, mask, ttids)
                preds.extend(torch.argmax(logits, -1).cpu().numpy())
                labels.extend(batch['labels'].numpy())
        acc = accuracy_score(labels, preds)
        mf1 = f1_score(labels, preds, average='macro', zero_division=0)
        print(f"{b} | n={len(sub):4d} | Acc={acc*100:.2f}% | Macro-F1={mf1:.4f}")

    reset_log(log)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",        nargs="+", choices=["base", "large"],
                        default=["base", "large"])
    parser.add_argument("--epochs",        type=int,  default=EPOCHS)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-q4",       action="store_true")
    parser.add_argument("--eval-only",     action="store_true")
    args = parser.parse_args()

    print(f"Device: {device}\n")
    train_df, val_df, test_df = load_data()

    if not args.skip_baseline:
        run_majority_baseline(test_df)

    all_results = {}

    for variant in args.models:
        cfg        = CONFIGS[variant]
        model_name = cfg["model_name"]
        dtype      = cfg["dtype"]
        model_path = os.path.join(MODEL_DIR, f"bert_{variant}_safety_model.pt")
        log_name   = f"bert_{variant}.txt"

        log = set_log(log_name)
        print(f"=== {model_name} ===\n")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        val_ds  = SafetyDataset(val_df,  tokenizer)
        test_ds = SafetyDataset(test_df, tokenizer)

        if not args.eval_only:
            train_ds = SafetyDataset(train_df, tokenizer)
            model    = BERTClassifier(model_name=model_name, dtype=dtype)
            train_model(model, train_ds, val_ds, cfg, args.epochs, model_path)
            del model; torch.cuda.empty_cache()

        if os.path.exists(model_path):
            print(f"\n--- Evaluare test set ({variant}) ---")
            acc, mf1 = evaluate_model(model_path, test_ds, model_name, dtype,
                                      bs=cfg["batch_size"])
            all_results[variant] = (acc, mf1)
        else:
            print(f"Model {variant} nu există la {model_path}, skip evaluare.")

        reset_log(log)

        if not args.skip_q4 and os.path.exists(model_path):
            run_q4(model_path, test_df, tokenizer, model_name, dtype, variant,
                   bs=cfg["batch_size"])

    if all_results:
        log = set_log("bert_summary.txt")
        print("=== Sumar BERT ===\n")
        print(f"{'Variant':<10} | {'Accuracy':>9} | {'Macro-F1':>9}")
        print("-" * 36)
        for v, (acc, mf1) in all_results.items():
            print(f"{v:<10} | {acc*100:>8.2f}% | {mf1:>9.4f}")
        reset_log(log)

        summary = {
            "models_tested": args.models,
            "max_length": MAX_LENGTH,
            "epochs": args.epochs,
            "timestamp": str(datetime.datetime.now()),
            "results": {
                v: {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}
                for v, (acc, mf1) in all_results.items()
            },
            "best_variant": max(all_results, key=lambda v: all_results[v][1])
        }
        summary_path = os.path.join(LOG_DIR, "bert_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✅ Summary JSON salvat: {summary_path}")
        best = summary['best_variant']
        print(f"🏆 Best: bert-{best} (Macro-F1={all_results[best][1]:.4f})")

    print("\n✅ Toate experimentele BERT finalizate. Loguri în ~/project/logs/")

if __name__ == "__main__":
    main()
