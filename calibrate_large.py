# ==============================
# Script 3: Temperature Calibration (DeBERTa-v3-BASE)
# Run AFTER vitaminc_finetune_base.py
# Fits temperature T on MNLI validation set
# Saves calibration.json to ./nli_deberta_base_vitaminc/
# RTX 4060 (8GB) optimized
# ==============================

import torch
import torch.nn as nn
from torch import optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import json

# ==============================
# Temperature Scaling
# ==============================
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature


# ==============================
# ECE computation
# ==============================
def compute_ece(logits, labels, n_bins=10):
    probs = torch.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)

    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].float().mean()
            bin_conf = confidences[mask].mean()
            ece += mask.float().mean() * (bin_conf - bin_acc).abs()
    return ece.item()


if __name__ == "__main__":

    MODEL_PATH = "./nli_deberta_base_vitaminc"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32     # base model — plenty of headroom on 8GB for inference
    MAX_LENGTH = 128    # matches training scripts

    print(f"Device: {DEVICE}")

    # ==============================
    # 1. Load MNLI matched validation
    # ==============================
    print("Loading MNLI validation set...")
    mnli = load_dataset("multi_nli")
    val = mnli["validation_matched"].filter(lambda x: x["label"] != -1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    # ==============================
    # 2. Print and verify label mapping
    # CRITICAL — this determines ENTAILMENT_INDEX in summac_zs_base.py
    # ==============================
    print("\nModel label mapping (VERIFY THIS):")
    print(model.config.id2label)
    print("Expected: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}\n")

    # ==============================
    # 3. Tokenize MNLI val
    # ==============================
    def tokenize(example):
        return tokenizer(
            example["premise"],
            example["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH
        )

    val_tokenized = val.map(tokenize, batched=True)
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_loader = DataLoader(val_tokenized, batch_size=BATCH_SIZE)

    # ==============================
    # 4. Collect logits
    # ==============================
    all_logits = []
    all_labels = []

    print("Collecting logits on MNLI val...")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits.cpu())
            all_labels.append(labels)

            if i % 20 == 0:
                print(f"  Batch {i}/{len(val_loader)}")

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    print(f"Collected {len(all_logits)} logits")

    # ==============================
    # 5. Fit temperature
    # ==============================
    ts = TemperatureScaling()
    optimizer = optim.LBFGS([ts.temperature], lr=0.01, max_iter=100)
    nll = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        loss = nll(ts(all_logits), all_labels)
        loss.backward()
        return loss

    print("Fitting temperature...")
    optimizer.step(eval_step)

    T = ts.temperature.item()
    print(f"\nOptimal Temperature T = {T:.4f}")

    # ==============================
    # 6. Gate check
    # ==============================
    print("\n--- GATE CHECK ---")
    if T < 1.0:
        print(f"⚠ T = {T:.4f} — model underconfident, using T=1.0")
        T = 1.0
    elif T > 3.0:
        print(f"✗ T = {T:.4f} — badly overconfident. Reduce label_smoothing to 0.02 and retrain.")
    elif 1.1 <= T <= 2.5:
        print(f"✓ T = {T:.4f} — within expected range [1.1, 2.5]")
    else:
        print(f"~ T = {T:.4f} — slightly outside range, acceptable")

    # ==============================
    # 7. ECE before and after
    # ==============================
    ece_before = compute_ece(all_logits, all_labels)
    ece_after = compute_ece(ts(all_logits), all_labels)

    print(f"\nECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}  (target <= 0.05)")

    if ece_after <= 0.05:
        print("✓ ECE gate passed")
    else:
        print("✗ ECE above 0.05 — calibration incomplete, check label smoothing")

    # ==============================
    # 8. Save calibration.json
    # ==============================
    calibration = {
        "temperature": T,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "model_label_mapping": {str(k): v for k, v in model.config.id2label.items()}
    }

    out_path = f"{MODEL_PATH}/calibration.json"
    with open(out_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\nSaved calibration.json to {out_path}")
    print("Contents:", json.dumps(calibration, indent=2))