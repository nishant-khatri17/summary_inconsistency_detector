# ==============================
# new_script.py
# Evaluate on remaining SummaC datasets: factcc, summeval, frank
# Same strategies as improved_score.py
# Results saved to ./new_results.json
# ==============================

import os
import sys
import json
import torch
import numpy as np
import argparse
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

# Fix Windows console encoding (cp1252 can't handle Unicode symbols)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ==============================
# Args
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument("--model",         type=str,   default="./nli_deberta_base_vitaminc")
parser.add_argument("--summac_folder", type=str,   default="./summac_benchmark")
parser.add_argument("--batch_size",    type=int,   default=8)
parser.add_argument("--max_len",       type=int,   default=128)

# SIFiD filtering
parser.add_argument("--beta",          type=float, default=0.0,
                    help="Relevance threshold for doc sentence filtering.")
parser.add_argument("--window",        type=int,   default=1,
                    help="Context window around each retained doc sentence.")

# CMWS / CWCS parameters
parser.add_argument("--margin_weight", type=float, default=0.8)
parser.add_argument("--lambda_c",      type=float, default=0.25)
parser.add_argument("--con_threshold", type=float, default=0.25)
parser.add_argument("--claim_temp",    type=float, default=5.0)
parser.add_argument("--softmin_alpha", type=float, default=3.0)
parser.add_argument("--topk",          type=int,   default=3)
parser.add_argument("--blend_w",       type=float, default=0.5)

parser.add_argument("--checkpoint",    type=str,   default="./new_checkpoint.json")
parser.add_argument("--output",        type=str,   default="./new_results.json")
args = parser.parse_args()

# ==============================
# Device
# ==============================
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ==============================
# Load NLI Model
# ==============================
print(f"\nLoading model: {args.model}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
model     = AutoModelForSequenceClassification.from_pretrained(args.model).to(DEVICE)
model.eval()

id2label            = model.config.id2label
ENTAILMENT_INDEX    = None
NEUTRAL_INDEX       = None
CONTRADICTION_INDEX = None

for idx, lbl in id2label.items():
    l = lbl.lower()
    if l in ("entailment", "supports"):
        ENTAILMENT_INDEX = int(idx)
    elif l == "neutral":
        NEUTRAL_INDEX = int(idx)
    elif l in ("contradiction", "refutes"):
        CONTRADICTION_INDEX = int(idx)

if ENTAILMENT_INDEX is None:
    raise ValueError(f"Cannot find entailment label in {id2label}")

has_neutral       = NEUTRAL_INDEX is not None
has_contradiction = CONTRADICTION_INDEX is not None

if not has_neutral:
    print("Warning: No neutral label. Disabling margin weighting.")
    args.margin_weight = 0.0
if not has_contradiction:
    print("Warning: No contradiction label. Disabling contradiction penalty.")
    args.lambda_c = 0.0

print(f"Label mapping:       {id2label}")
print(f"Entailment index:    {ENTAILMENT_INDEX}")
print(f"Neutral index:       {NEUTRAL_INDEX}")
print(f"Contradiction index: {CONTRADICTION_INDEX}")
print(f"beta (filter):       {args.beta}")
print(f"window:              {args.window}")
print(f"margin_weight:       {args.margin_weight}")
print(f"lambda_c:            {args.lambda_c}")
print(f"claim_temp:          {args.claim_temp}")
print(f"softmin_alpha:       {args.softmin_alpha}")
print(f"topk:                {args.topk}")
print(f"blend_w:             {args.blend_w}\n")


# ==============================
# Full NLI Matrix
# ==============================
def build_full_nli_matrix(doc_sents, sum_sents):
    M, N = len(doc_sents), len(sum_sents)
    E    = torch.zeros(M, N)
    Neu  = torch.zeros(M, N)
    C    = torch.zeros(M, N)

    pairs = [(doc_sents[i], sum_sents[j])
             for i in range(M) for j in range(N)]

    for start in range(0, len(pairs), args.batch_size):
        batch      = pairs[start: start + args.batch_size]
        premises   = [p[0] for p in batch]
        hypotheses = [p[1] for p in batch]

        inputs = tokenizer(
            premises, hypotheses,
            truncation=True, padding=True,
            max_length=args.max_len,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=-1).cpu()

        for k in range(len(batch)):
            flat = start + k
            i, j = flat // N, flat % N
            E[i, j]   = probs[k, ENTAILMENT_INDEX]
            if has_neutral:
                Neu[i, j] = probs[k, NEUTRAL_INDEX]
            if has_contradiction:
                C[i, j]   = probs[k, CONTRADICTION_INDEX]

    return E, Neu, C


# ==============================
# SIFiD-style Document Sentence Filter
# ==============================
def sifid_mask(E, C, beta=0.0, window=1):
    M = E.shape[0]
    net        = E - C
    relevance  = net.max(dim=1).values

    relevant = relevance > beta

    expanded = relevant.clone()
    for i in range(M):
        if relevant[i]:
            for w in range(-window, window + 1):
                j = i + w
                if 0 <= j < M:
                    expanded[j] = True

    if expanded.sum() == 0:
        expanded = torch.ones(M, dtype=torch.bool)

    return expanded


# ==============================
# Aggregation Strategies
# ==============================

def agg_zs(E, Neu, C):
    return E.max(dim=0).values.mean().item()

def agg_sifid_zs(E, Neu, C):
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f  = E[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return E_f.max(dim=0).values.mean().item()

def agg_cwcs(E, Neu, C):
    best_ent = E.max(dim=0).values
    best_con = C.max(dim=0).values
    claim_strength = torch.max(best_ent, best_con)

    margin     = torch.relu(E - Neu)
    agg_w      = E + args.margin_weight * margin
    agg_s      = torch.softmax(agg_w * 10.0, dim=0)
    margin_score = (agg_s * E).sum(dim=0)

    support = 0.5 * best_ent + 0.5 * margin_score

    if has_contradiction:
        con_spread = (C > args.con_threshold).float().mean(dim=0)
        support    = support - args.lambda_c * con_spread
        support    = support.clamp(0.0, 1.0)

    claim_w = torch.softmax(claim_strength * args.claim_temp, dim=0)
    return (claim_w * support).sum().item()

def agg_sifid_cwcs(E, Neu, C):
    mask  = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f, Neu_f, C_f = E[mask], Neu[mask], C[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return agg_cwcs(E_f, Neu_f, C_f)

def agg_sifid_softmin(E, Neu, C):
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f  = E[mask]
    if E_f.shape[0] == 0:
        return 0.5
    col_max  = E_f.max(dim=0).values
    alpha    = args.softmin_alpha
    soft_min = -1.0 / alpha * torch.log(
        torch.mean(torch.exp(-alpha * col_max))
    )
    return float(soft_min.clamp(0.0, 1.0).item())

def agg_net_entailment(E, Neu, C):
    net = E - C
    col_max = net.max(dim=0).values
    return col_max.clamp(0.0, 1.0).mean().item()

def agg_scaled_entailment(E, Neu, C):
    scaled = E * (1.0 - C)
    col_max = scaled.max(dim=0).values
    return col_max.mean().item()

def agg_topk_mean(E, Neu, C):
    k = min(args.topk, E.shape[0])
    topk_vals = E.topk(k, dim=0).values
    col_scores = topk_vals.mean(dim=0)
    return col_scores.mean().item()

def agg_row_normalized(E, Neu, C):
    M, N = E.shape
    if N <= 1:
        return agg_zs(E, Neu, C)
    row_mean = E.mean(dim=1, keepdim=True)
    row_std  = E.std(dim=1, keepdim=True).clamp(min=1e-6)
    E_z      = (E - row_mean) / row_std
    col_max  = E_z.max(dim=0).values
    scores   = torch.sigmoid(col_max)
    return scores.mean().item()

def agg_hybrid_blend(E, Neu, C):
    w = args.blend_w
    zs_score  = agg_zs(E, Neu, C)
    net_score = agg_net_entailment(E, Neu, C)
    return w * zs_score + (1.0 - w) * net_score

def agg_sifid_net_ent(E, Neu, C):
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f, Neu_f, C_f = E[mask], Neu[mask], C[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return agg_net_entailment(E_f, Neu_f, C_f)

def agg_sifid_scaled_ent(E, Neu, C):
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f, Neu_f, C_f = E[mask], Neu[mask], C[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return agg_scaled_entailment(E_f, Neu_f, C_f)


# ==============================
# Strategy Registry
# ==============================
STRATEGIES = {
    "zs_max"           : agg_zs,
    "sifid_zs"         : agg_sifid_zs,
    "cwcs"             : agg_cwcs,
    "sifid_cwcs"       : agg_sifid_cwcs,
    "sifid_softmin"    : agg_sifid_softmin,
    "net_entailment"   : agg_net_entailment,
    "scaled_ent"       : agg_scaled_entailment,
    "topk_mean"        : agg_topk_mean,
    "row_normalized"   : agg_row_normalized,
    "hybrid_blend"     : agg_hybrid_blend,
    "sifid_net_ent"    : agg_sifid_net_ent,
    "sifid_scaled_ent" : agg_sifid_scaled_ent,
}


# ==============================
# Score a single (document, summary) pair
# ==============================
def score_pair(document, summary):
    doc_sents = sent_tokenize(document)
    sum_sents = sent_tokenize(summary)

    if not doc_sents or not sum_sents:
        return {k: 0.5 for k in STRATEGIES}

    E, Neu, C = build_full_nli_matrix(doc_sents, sum_sents)
    return {name: fn(E, Neu, C) for name, fn in STRATEGIES.items()}


# ==============================
# Threshold Tuning
# ==============================
def tune_threshold(scores, labels):
    best_thresh, best_bacc = 0.5, 0.0
    for thresh in np.linspace(0.05, 0.95, 91):
        preds = [1 if s >= thresh else 0 for s in scores]
        bacc  = balanced_accuracy_score(labels, preds)
        if bacc > best_bacc:
            best_bacc, best_thresh = bacc, thresh
    return best_thresh, best_bacc


# ==============================
# Checkpoint helpers
# ==============================
def load_checkpoint(name):
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            data = json.load(f)
        if data.get("dataset_name") == name:
            all_scores = data.get("all_scores", {})
            missing = [s for s in STRATEGIES if s not in all_scores]
            if missing:
                print(f"  Checkpoint missing strategies {missing}, starting fresh.")
                return {strat: [] for strat in STRATEGIES}, []
            return all_scores, data.get("labels", [])
    return {strat: [] for strat in STRATEGIES}, []


def save_checkpoint(name, all_scores, labels):
    with open(args.checkpoint, "w") as f:
        json.dump({
            "dataset_name": name,
            "all_scores"  : all_scores,
            "labels"      : labels,
        }, f)


def clear_checkpoint():
    if os.path.exists(args.checkpoint):
        os.remove(args.checkpoint)


# ==============================
# Sanity Check
# ==============================
def sanity_check():
    print("=" * 70)
    print("SANITY CHECK — Scoring strategies on new datasets")
    print("=" * 70)

    document = (
        "The company reported a revenue of 5 billion dollars in Q3. "
        "The CEO announced plans to expand into Asian markets next year. "
        "Employee headcount grew by 12 percent compared to last quarter."
    )
    consistent   = "The company's Q3 revenue reached 5 billion dollars."
    inconsistent = "The company reported a revenue of 9 billion dollars in Q3."
    hallucinated = "The company plans to lay off 30 percent of its workforce."

    s_c = score_pair(document, consistent)
    s_i = score_pair(document, inconsistent)
    s_h = score_pair(document, hallucinated)

    print(f"\n{'Strategy':<16}  {'Consistent':>10}  {'Inconsistent':>12}  "
          f"{'Hallucinated':>12}  {'Pass':>5}")
    print("-" * 65)

    all_passed = True
    for name in STRATEGIES:
        c, i, h = s_c[name], s_i[name], s_h[name]
        passed     = (c > i) and (c > h)
        all_passed = all_passed and passed
        print(f"{name:<16}  {c:>10.4f}  {i:>12.4f}  {h:>12.4f}  "
              f"{'Y' if passed else 'N':>5}")

    print("=" * 70)
    print("All strategies passed sanity check\n" if all_passed
          else "Some strategies FAILED -- check model\n")
    return all_passed


# ==============================
# Score a single dataset
# ==============================
def score_dataset(name, data, doc_key, claim_key, label_fn):
    print(f"\n{'='*70}")
    print(f"Dataset: {name}  ({len(data)} samples)")

    all_scores, labels = load_checkpoint(name)
    start_idx = len(labels)

    if start_idx > 0:
        print(f"  Resuming from checkpoint at sample {start_idx}")

    for i in range(start_idx, len(data)):
        item = data[i]
        try:
            pair_scores = score_pair(item[doc_key], item[claim_key])
        except Exception as e:
            print(f"  Warning: error on sample {i}: {e}. Using 0.5.")
            pair_scores = {k: 0.5 for k in STRATEGIES}

        for strat in STRATEGIES:
            all_scores[strat].append(pair_scores[strat])
        labels.append(label_fn(item))

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(data)}")
            save_checkpoint(name, all_scores, labels)

    clear_checkpoint()

    if len(set(labels)) < 2:
        print(f"  Skipping {name} -- only one class present")
        return None

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"  Class balance:  {n_pos} consistent / {n_neg} inconsistent\n")
    print(f"  {'Strategy':<16}  {'BAcc':>6}  {'AUC':>6}  {'Thresh':>7}")
    print(f"  {'-'*42}")

    best_strat, best_bacc, best_thresh, best_auc = None, 0.0, 0.5, float("nan")
    strategy_results = {}

    for strat in STRATEGIES:
        scores = all_scores[strat]
        thresh, bacc = tune_threshold(scores, labels)
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = float("nan")

        is_best = bacc > best_bacc
        star    = "*" if is_best else " "
        print(f"  {star}{strat:<15}  {bacc:.4f}  "
              f"{auc if not np.isnan(auc) else float('nan'):.4f}  {thresh:.2f}")

        strategy_results[strat] = {
            "balanced_accuracy": bacc,
            "roc_auc"          : auc,
            "threshold"        : thresh,
        }

        if is_best:
            best_bacc, best_thresh, best_auc, best_strat = bacc, thresh, auc, strat

    zs_bacc = strategy_results["zs_max"]["balanced_accuracy"]
    delta   = best_bacc - zs_bacc
    print(f"\n  Best: {best_strat}  BAcc={best_bacc:.4f}  AUC={best_auc:.4f}")
    print(f"  vs ZS baseline: {'+' if delta >= 0 else ''}{delta:.4f}  "
          f"({'IMPROVED' if delta > 0 else 'below ZS'})")

    return {
        "best_strategy"    : best_strat,
        "balanced_accuracy": best_bacc,
        "roc_auc"          : best_auc,
        "threshold"        : best_thresh,
        "n_samples"        : len(data),
        "n_consistent"     : n_pos,
        "n_inconsistent"   : n_neg,
        "per_strategy"     : strategy_results,
    }


# ==============================
# Evaluate on remaining SummaC datasets
# ==============================
def evaluate_new_datasets():
    try:
        from summac.benchmark import SummaCBenchmark
    except ImportError:
        print("Run: pip install summac")
        return

    TARGET_DATASETS = ["frank"]

    print(f"Loading SummaC benchmark (frank) from {args.summac_folder}...")
    benchmark = SummaCBenchmark(
        benchmark_folder=args.summac_folder,
        dataset_names=TARGET_DATASETS,
        cut="val"
    )
    print(f"Loaded:  {[d['name'] for d in benchmark.datasets]}\n")

    all_results = {}
    best_baccs  = []
    zs_baccs    = []

    for dataset_info in benchmark.datasets:
        name = dataset_info["name"]

        result = score_dataset(
            name, dataset_info["dataset"],
            doc_key  = "document",
            claim_key= "claim",
            label_fn = lambda item: item["label"]
        )
        if result is None:
            continue

        all_results[name] = result
        best_baccs.append(result["balanced_accuracy"])
        zs_baccs.append(result["per_strategy"]["zs_max"]["balanced_accuracy"])

    if not all_results:
        print("No valid results.")
        return

    overall_best = float(np.mean(best_baccs))
    overall_zs   = float(np.mean(zs_baccs))
    valid_aucs   = [r["roc_auc"] for r in all_results.values()
                    if not np.isnan(r["roc_auc"])]
    overall_auc  = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    # === Final Summary Table ===
    print(f"\n{'='*70}")
    print("RESULTS -- Frank Dataset")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<14}  {'ZS BAcc':>8}  {'Best BAcc':>10}  "
          f"{'Best Strategy':>16}  {'Delta':>7}")
    print(f"  {'-'*62}")

    for name, res in all_results.items():
        zs_b   = res["per_strategy"]["zs_max"]["balanced_accuracy"]
        best_b = res["balanced_accuracy"]
        d      = best_b - zs_b
        print(f"  {name:<14}  {zs_b:>8.4f}  {best_b:>10.4f}  "
              f"{res['best_strategy']:>16}  "
              f"{'+' if d >= 0 else ''}{d:>6.4f}")

    print(f"  {'-'*62}")
    delta = overall_best - overall_zs
    print(f"  {'Average':<14}  {overall_zs:>8.4f}  {overall_best:>10.4f}  "
          f"{'':>16}  {'+' if delta >= 0 else ''}{delta:>6.4f}")

    print(f"\n  Datasets evaluated:         {len(all_results)}")
    print(f"  ZS baseline (this run):     {overall_zs:.4f}")
    print(f"  Best improved score:        {overall_best:.4f}")
    print(f"  Best ROC-AUC:               {overall_auc:.4f}")

    # === Strategy-level analysis ===
    print(f"\n  {'-'*42}")
    print("  Per-Strategy Average BAcc across new datasets:")
    for strat in STRATEGIES:
        avg = np.mean([r["per_strategy"][strat]["balanced_accuracy"]
                       for r in all_results.values()])
        marker = " <-best" if strat == max(
            STRATEGIES,
            key=lambda s: np.mean([r["per_strategy"][s]["balanced_accuracy"]
                                   for r in all_results.values()])
        ) else ""
        print(f"    {strat:<16}  avg BAcc = {avg:.4f}{marker}")

    with open(args.output, "w") as f:
        json.dump({
            "model"          : args.model,
            "scorer"         : "SIFiD+CWCS Improved",
            "evaluated_on"   : "frank",
            "params"         : {
                "beta"          : args.beta,
                "window"        : args.window,
                "margin_weight" : args.margin_weight,
                "lambda_c"      : args.lambda_c,
                "con_threshold" : args.con_threshold,
                "claim_temp"    : args.claim_temp,
                "softmin_alpha" : args.softmin_alpha,
                "topk"          : args.topk,
                "blend_w"       : args.blend_w,
            },
            "n_datasets"         : len(all_results),
            "overall_best_bacc"  : overall_best,
            "overall_zs_bacc"    : overall_zs,
            "overall_roc_auc"    : overall_auc,
            "delta_vs_zs"        : delta,
            "per_dataset"        : all_results,
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    passed = sanity_check()
    if not passed:
        print("Sanity check failed -- fix model before benchmarking.")
    else:
        print("Proceed to evaluate factcc, summeval, frank? (y/n): ", end="")
        if input().strip().lower() == "y":
            evaluate_new_datasets()
        else:
            print("Skipping.")
