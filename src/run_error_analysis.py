"""
Rulare:
    source ~/pan_venv/bin/activate
    python3 ~/project/scripts/run_error_analysis.py
    python3 ~/project/scripts/run_error_analysis.py --rank 16   # alt rank
    python3 ~/project/scripts/run_error_analysis.py --n-examples 20
"""

import os, sys, json, argparse, textwrap, datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')   # fără display GUI pe VM
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score)
from collections import Counter
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────
BASE       = os.path.expanduser("~/project/data")
MODEL_DIR  = os.path.expanduser("~/project/model")
LOG_DIR    = os.path.expanduser("~/project/logs")
MODEL_NAME = "mistralai/Mistral-3B-Instruct-v0.3"
MAX_LENGTH = 1024
BATCH_SIZE = 4

os.makedirs(LOG_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"safe": 0, "potentially unsafe": 1, "unsafe": 2}
ID2LABEL  = {0: "safe", 1: "potentially_unsafe", 2: "unsafe"}
CLASS_NAMES = ["safe", "potentially_unsafe", "unsafe"]

# ─────────────────────────────────────────────────────────────────────────────
# DATE
# ─────────────────────────────────────────────────────────────────────────────
def load_test_data():
    dv_unsafe      = pd.read_json(f"{BASE}/Validation/valid_unsafe.jsonl",             lines=True)
    dv_safe        = pd.read_json(f"{BASE}/Validation/valid_safe.jsonl",               lines=True)
    dv_potentially = pd.read_json(f"{BASE}/Validation/valid_potentially_unsafe.jsonl", lines=True)
    test_df        = pd.concat([dv_unsafe, dv_safe, dv_potentially], ignore_index=True)
    print(f"Test set: {len(test_df)} exemple")
    print(test_df['label'].value_counts())
    return test_df


class SafetyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=MAX_LENGTH):
        self.data      = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        query = str(row.get('query', ''))
        trace = str(row.get('reasoning_trace', ''))
        text  = f"Query: {query}\nReasoning Trace:\n{trace}"
        enc   = self.tokenizer(
            text, truncation=True, max_length=self.max_len,
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
# MODEL — identic cu run_all_mistral.py
# ─────────────────────────────────────────────────────────────────────────────
class SafetyClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=3, r_value=64):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        lora_cfg = LoraConfig(
            r=r_value, lora_alpha=2 * r_value,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1, bias="none", task_type="FEATURE_EXTRACTION"
        )
        self.base_model = get_peft_model(self.base_model, lora_cfg)
        hidden = self.base_model.config.hidden_size  # 3072
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out  = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        lhs  = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(lhs.size()).float()
        vec  = torch.sum(lhs * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return self.classifier(vec.float())


# ─────────────────────────────────────────────────────────────────────────────
# INFERENȚĂ — returnează predicții + probabilități + indecși originali
# ─────────────────────────────────────────────────────────────────────────────
def run_inference(model, test_ds):
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels, all_probs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferență"):
            ids    = batch['input_ids'].to(device)
            mask   = batch['attention_mask'].to(device)
            lbls   = batch['labels']
            logits = model(ids, mask)
            probs  = torch.softmax(logits.float(), dim=-1)
            preds  = torch.argmax(probs, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(lbls.numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    return all_preds, all_labels, all_probs


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFUSION MATRIX — PNG + JSON
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, rank, save_dir=LOG_DIR):
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])

    # Normalizată pe rând (recall per clasă)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — Mistral-3B + LoRA r={rank}", fontsize=14, fontweight='bold')

    # Stânga: counts absolute
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Counts absolute")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Ground Truth")

    # Dreapta: normalizată
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
    axes[1].set_title("Normalized (row = recall per clasă)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Ground Truth")

    plt.tight_layout()
    png_path = os.path.join(save_dir, f"confusion_matrix_mistral_r{rank}.png")
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix salvată: {png_path}")

    # Salvăm și numerele în JSON pentru paper
    cm_data = {
        "model": f"Mistral-3B + LoRA r={rank}",
        "classes": CLASS_NAMES,
        "confusion_matrix_counts": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist(),
        "per_class": {}
    }
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        cm_data["per_class"][cls] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4)
        }

    json_path = os.path.join(save_dir, f"confusion_matrix_mistral_r{rank}.json")
    with open(json_path, "w") as f:
        json.dump(cm_data, f, indent=2)
    print(f"✅ Confusion matrix JSON: {json_path}")

    # Print sumar în consolă
    print(f"\n{'='*50}")
    print("CONFUSION MATRIX — valori absolute")
    print(f"{'':25} {'safe':>12} {'pot_unsafe':>12} {'unsafe':>12}")
    for i, cls in enumerate(CLASS_NAMES):
        row = "  ".join(f"{cm[i,j]:12d}" for j in range(3))
        print(f"  GT={cls:<20} {row}")
    print(f"\n{'='*50}")
    print("PER-CLASS METRICS")
    print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)
    for cls, v in cm_data["per_class"].items():
        print(f"  {cls:<20} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f} "
              f"{v['TP']:>6} {v['FP']:>6} {v['FN']:>6}")

    return cm_data


# ─────────────────────────────────────────────────────────────────────────────
# 2. ERROR ANALYSIS — FP/FN calitativ
# ─────────────────────────────────────────────────────────────────────────────
def truncate_text(text, max_chars=300):
    """Trunchiază textul pentru afișare, păstrând începutul și sfârșitul."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n  [...]\n  " + text[-half:]


def auto_explain(gt_label, pred_label, query, trace, probs):
    """
    Generează o explicație automată bazată pe pattern-uri comune de eroare.
    Aceasta e o euristică — pentru paper vei completa manual cu observații specifice.
    """
    gt   = CLASS_NAMES[gt_label]
    pred = CLASS_NAMES[pred_label]
    conf = max(probs)
    q_lower = query.lower()
    t_lower = trace.lower()

    hints = []

    # Pattern 1: FP — unsafe prezis ca safe
    if gt_label == 2 and pred_label == 0:
        hints.append("MODEL IGNORES QUERY SIGNAL: modelul se concentrează pe pașii de raționament "
                     "care pot părea neutri, ignorând intenția periculoasă din query.")
        if any(w in t_lower for w in ["however", "but", "should not", "it's important", "caution"]):
            hints.append("HEDGING LANGUAGE: traiectoria conține limbaj de avertizare ('however', "
                         "'should not') care mimează un raționament sigur, confuzând clasificatorul.")

    # Pattern 2: FN — safe prezis ca unsafe/pot_unsafe
    elif gt_label == 0 and pred_label in [1, 2]:
        hints.append("FALSE ALARM: query-ul sau trace-ul conțin vocabular asociat cu siguranța "
                     "(ex: domenii medicale, juridice, de securitate) fără intenție reală de vătămare.")
        if any(w in q_lower for w in ["how to", "what is", "explain", "describe"]):
            hints.append("INFORMATIONAL QUERY: întrebarea e educațională/informativă, dar topicul "
                         "(ex: arme, substanțe) activează false positive.")

    # Pattern 3: potentially_unsafe confundat cu safe
    elif gt_label == 1 and pred_label == 0:
        hints.append("AMBIGUITY COLLAPSE: clasa potentially_unsafe e cea mai dificilă — limita "
                     "față de safe e semantic ambiguă. Modelul colapsează spre majority class (safe).")
        hints.append(f"Confidence pentru safe: {probs[0]:.2f} vs pot_unsafe: {probs[1]:.2f} — "
                     f"diferență mică sugerează că modelul e nesigur.")

    # Pattern 4: potentially_unsafe confundat cu unsafe
    elif gt_label == 1 and pred_label == 2:
        hints.append("OVER-CLASSIFICATION: modelul supraestimează pericolul. Traiectoria conține "
                     "indicii de risc dar nu furnizează instrucțiuni acționabile.")

    # Pattern 5: unsafe confundat cu potentially_unsafe
    elif gt_label == 2 and pred_label == 1:
        hints.append("UNDER-CLASSIFICATION: conținut clar periculos clasificat ca 'potențial'. "
                     "Posibil că pașii periculoși apar târziu în traiectorie (diluat de mean pooling).")

    if conf > 0.85:
        hints.append(f"HIGH CONFIDENCE ERROR (conf={conf:.2f}): eroare cu încredere mare — "
                     f"modelul e sigur de predicția greșită, semn de overfitting pe pattern superficial.")
    elif conf < 0.55:
        hints.append(f"LOW CONFIDENCE (conf={conf:.2f}): modelul ezită — exemplu de graniță.")

    return hints if hints else ["Eroare fără pattern clar identificat."]


def extract_error_cases(test_df, preds, labels, probs, n_examples=10):
    """
    Extrage FP și FN pentru fiecare combinație de clase greșite.
    Returnează dict cu tipuri de eroare și exemple.
    """
    errors = {}

    for gt in range(3):
        for pred in range(3):
            if gt == pred:
                continue
            key = f"GT={CLASS_NAMES[gt]}__PRED={CLASS_NAMES[pred]}"
            indices = [i for i, (g, p) in enumerate(zip(labels, preds)) if g == gt and p == pred]
            errors[key] = {
                "count": len(indices),
                "gt_label": CLASS_NAMES[gt],
                "pred_label": CLASS_NAMES[pred],
                "examples": []
            }
            # Sortăm după confidence descrescător — erorile cu confidence mare sunt mai interesante
            sorted_indices = sorted(indices, key=lambda i: max(probs[i]), reverse=True)

            for idx in sorted_indices[:n_examples]:
                row   = test_df.iloc[idx]
                query = str(row.get('query', ''))
                trace = str(row.get('reasoning_trace', ''))
                prob  = probs[idx]
                hints = auto_explain(gt, pred, query, trace, prob)

                errors[key]["examples"].append({
                    "index":       idx,
                    "query":       query,
                    "trace_snippet": trace[:500] + ("..." if len(trace) > 500 else ""),
                    "gt_label":    CLASS_NAMES[gt],
                    "pred_label":  CLASS_NAMES[pred],
                    "confidence":  round(max(prob), 4),
                    "probs":       {CLASS_NAMES[i]: round(prob[i], 4) for i in range(3)},
                    "auto_hints":  hints
                })

    return errors


def save_error_analysis_txt(errors, rank, acc, macro_f1, save_dir=LOG_DIR):
    """Salvează raportul text formatat pentru citire directă."""
    path = os.path.join(save_dir, f"error_analysis_mistral_r{rank}.txt")
    sep  = "=" * 70

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n")
        f.write(f"ERROR ANALYSIS — Mistral-3B + LoRA r={rank}\n")
        f.write(f"Generat: {datetime.datetime.now()}\n")
        f.write(f"Test set accuracy: {acc*100:.2f}%  |  Macro-F1: {macro_f1:.4f}\n")
        f.write(f"{sep}\n\n")

        # Sumar erori
        f.write("SUMAR ERORI PER TIP:\n")
        total_errors = sum(v["count"] for v in errors.values())
        for key, v in sorted(errors.items(), key=lambda x: x[1]["count"], reverse=True):
            pct = v["count"] / total_errors * 100 if total_errors > 0 else 0
            f.write(f"  {key:<45} : {v['count']:4d} exemple ({pct:.1f}%)\n")
        f.write(f"\n  TOTAL ERORI: {total_errors}\n\n")

        # Exemple calitative
        for key, v in errors.items():
            if v["count"] == 0:
                continue
            f.write(f"\n{sep}\n")
            f.write(f"TIP EROARE: GT={v['gt_label'].upper()} → PRED={v['pred_label'].upper()}\n")
            f.write(f"Total cazuri: {v['count']}\n")
            f.write(f"{sep}\n")

            for i, ex in enumerate(v["examples"], 1):
                f.write(f"\n--- Exemplu {i} (index={ex['index']}) ---\n")
                f.write(f"Confidence: {ex['confidence']:.4f}  |  ")
                f.write(f"Probs: safe={ex['probs']['safe']:.3f}, "
                        f"pot={ex['probs']['potentially_unsafe']:.3f}, "
                        f"unsafe={ex['probs']['unsafe']:.3f}\n\n")
                f.write(f"QUERY:\n  {textwrap.fill(ex['query'][:300], width=66, subsequent_indent='  ')}\n\n")
                f.write(f"TRACE (primele 500 chars):\n")
                for line in ex['trace_snippet'][:500].split('\n')[:8]:
                    f.write(f"  {line}\n")
                f.write(f"\nEXPLICAȚII AUTOMATE:\n")
                for hint in ex['auto_hints']:
                    f.write(f"  ▸ {textwrap.fill(hint, width=66, subsequent_indent='    ')}\n")
                f.write("\n")

    print(f"✅ Error analysis TXT: {path}")
    return path


def save_error_analysis_json(errors, rank, save_dir=LOG_DIR):
    path = os.path.join(save_dir, f"error_analysis_mistral_r{rank}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"✅ Error analysis JSON: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",       type=int, default=64,
                        help="LoRA rank al modelului salvat (default: 64)")
    parser.add_argument("--n-examples", type=int, default=10,
                        help="Număr de exemple FP/FN per tip de eroare (default: 10)")
    args = parser.parse_args()

    model_path = os.path.join(MODEL_DIR, f"mistral_safety_model_r{args.rank}.pt")
    if not os.path.exists(model_path):
        print(f"❌ Model nu găsit în {MODEL_DIR}")
        print(f"   Fișiere disponibile: {os.listdir(MODEL_DIR)}")
        sys.exit(1)

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"LoRA rank: {args.rank}\n")

    # 1. Încarcă date
    test_df = load_test_data()

    # 2. Încarcă model
    print("\nÎncărcare model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = SafetyClassifier(r_value=args.rank).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✅ Model încărcat\n")

    # 3. Inferență
    test_ds             = SafetyDataset(test_df, tokenizer)
    preds, labels, probs = run_inference(model, test_ds)

    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    print(f"\nAccuracy: {acc*100:.2f}%  |  Macro-F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, labels=[0,1,2],
                                 target_names=CLASS_NAMES, zero_division=0))

    # 4. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    cm_data = plot_confusion_matrix(labels, preds, rank=args.rank)

    # 5. Error Analysis
    print("\n--- Error Analysis ---")
    errors = extract_error_cases(test_df, preds, labels, probs,
                                  n_examples=args.n_examples)

    # Statistici rapide
    print("\nTipuri de eroare (sortate după frecvență):")
    total_errors = sum(v["count"] for v in errors.values())
    for key, v in sorted(errors.items(), key=lambda x: x[1]["count"], reverse=True):
        if v["count"] > 0:
            pct = v["count"] / total_errors * 100
            print(f"  {key:<45} : {v['count']:4d} ({pct:.1f}%)")

    save_error_analysis_txt(errors, args.rank, acc, macro_f1)
    save_error_analysis_json(errors, args.rank)

    print(f"\n✅ Toate fișierele salvate în {LOG_DIR}/")
    print(f"   - confusion_matrix_mistral_r{args.rank}.png")
    print(f"   - confusion_matrix_mistral_r{args.rank}.json")
    print(f"   - error_analysis_mistral_r{args.rank}.txt")
    print(f"   - error_analysis_mistral_r{args.rank}.json")


if __name__ == "__main__":
    main()
