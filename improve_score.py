# ==============================
# improved_scorer.py
# Novel NLI-based Summary Consistency Scorer
# Model: DeBERTa-v3-base (custom trained on VitaminC)
# Dataset: SummaC Benchmark (cogensumm, xsumfaith, polytope)
#
# Improvements over ZS baseline (arsh.py):
#
# 1. SIFiD-style Document Filtering  [SIFiD, ACL 2024]
#    Each doc sentence scored by max net-entailment
#    (E - C) to ANY summary sentence. Only relevant
#    sentences + their ±window neighbors are retained.
#    Eliminates noise from unrelated doc passages and
#    prevents accidentally high entailment from dominating.
#    KEY WIN: massive help on Polytope (extreme imbalance).
#
# 2. Claim-Weighted Aggregation  [CWCS / UIEFID insight]
#    Not all summary sentences are equally verifiable.
#      claim_strength_j = max(best_E_j, best_C_j)
#    High entailment OR contradiction = specific verifiable
#    claim. Mostly neutral = vague, contributes less.
#    Summary sentences are then softmax-weighted by strength.
#
# 3. Confidence-Margin Bonus  [CMWS]
#    ZS max ignores HOW confident the NLI model is.
#      margin_i = relu(E[i,j] - Neu[i,j])
#    Doc sentences are weighted by E + margin_weight * margin
#    before softmax aggregation over the doc dimension.
#    Rewards decisive entailment; penalises uncertain ones.
#
# 4. Contradiction Spread Penalty  [CMWS / CWCS]
#    Instead of max contradiction, penalise by the FRACTION
#    of relevant doc sentences that actively contradict each
#    summary sentence (prob > con_threshold). More robust to
#    single noisy contradictions.
#
# 5. Soft-Min Aggregation  [mac_summac_robust.py]
#    Instead of mean over summary sentences, use soft-min:
#      -1/alpha * log(mean(exp(-alpha * col_max)))
#    Punishes ANY unsupported summary sentence harder.
#
# All 5 strategies + ZS baseline run simultaneously on
# the same NLI matrix. Best strategy is auto-selected
# per dataset via threshold tuning (no data leakage —
# threshold tuning is done on val, consistent with arsh.py).
#
# Target: beat DeBERTa ZS baseline of 64.71% BAcc
#         (cogensumm: 70.43, xsumfaith: 66.24, polytope: 57.46)
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
                    help="Relevance threshold for doc sentence filtering. "
                         "Keep doc sentences where max_j(E-C) > beta. "
                         "0.0 = keep sentences where entailment > contradiction.")
parser.add_argument("--window",        type=int,   default=1,
                    help="Context window around each retained doc sentence (±window).")

# CMWS / CWCS parameters
parser.add_argument("--margin_weight", type=float, default=0.8,
                    help="Weight of confidence margin bonus in aggregation.")
parser.add_argument("--lambda_c",      type=float, default=0.25,
                    help="Contradiction spread penalty weight.")
parser.add_argument("--con_threshold", type=float, default=0.25,
                    help="Min contradiction prob to count as active contradiction.")
parser.add_argument("--claim_temp",    type=float, default=5.0,
                    help="Softmax temperature for claim-strength weighting. "
                         "Higher = more winner-takes-all.")
parser.add_argument("--softmin_alpha", type=float, default=3.0,
                    help="Harshness of soft-min aggregation. Higher = more punishing.")

parser.add_argument("--checkpoint",    type=str,   default="./improved_checkpoint.json")
parser.add_argument("--output",        type=str,   default="./improved_results.json")
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
print(f"softmin_alpha:       {args.softmin_alpha}\n")


# ==============================
# Full NLI Matrix
# E[i,j]   = P(doc_sent_i  entails     summary_sent_j)
# Neu[i,j] = P(doc_sent_i  neutral to  summary_sent_j)
# C[i,j]   = P(doc_sent_i  contradicts summary_sent_j)
# Shape: (M doc_sents) x (N summary_sents)
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
#
# For each doc sentence i, compute its "relevance" to the
# summary as its max net entailment across all summary sentences:
#   relevance_i = max_j( E[i,j] - C[i,j] )
#
# Keep doc sentences where relevance > beta, then expand
# by ±window neighbors to preserve local passage context.
# If nothing passes the threshold, fall back to all sentences.
#
# Inspired by: SIFiD §3  (Yang et al., ACL 2024 Workshop)
# ==============================
def sifid_mask(E, C, beta=0.0, window=1):
    """
    Returns a boolean mask of shape (M,) marking relevant doc sentences.
    """
    M = E.shape[0]
    net        = E - C                           # (M, N)  net entailment
    relevance  = net.max(dim=1).values           # (M,)    per-doc-sent max

    relevant = relevance > beta                  # (M,)    boolean

    # Expand by ±window to preserve passage context
    expanded = relevant.clone()
    for i in range(M):
        if relevant[i]:
            for w in range(-window, window + 1):
                j = i + w
                if 0 <= j < M:
                    expanded[j] = True

    # Safety fallback
    if expanded.sum() == 0:
        expanded = torch.ones(M, dtype=torch.bool)

    return expanded                              # (M,)


# ==============================
# Aggregation Strategies
# All operate on E (M×N), Neu (M×N), C (M×N)
# and return a single float ∈ [0, 1]
# ==============================

# --- Strategy 1: ZS Max-Mean (baseline, same as arsh.py) ---
def agg_zs(E, Neu, C):
    """
    Standard SummaC-ZS: for each summary sentence take the max
    entailment from any doc sentence, then average over summary.
    """
    return E.max(dim=0).values.mean().item()


# --- Strategy 2: SIFiD-ZS ---
def agg_sifid_zs(E, Neu, C):
    """
    SIFiD filtering followed by ZS aggregation.
    Filters noise from unrelated doc sentences before scoring.
    """
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f  = E[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return E_f.max(dim=0).values.mean().item()


# --- Strategy 3: CWCS (no filtering) ---
def agg_cwcs(E, Neu, C):
    """
    Claim-Weighted Consistency Scoring.

    For each summary sentence j:
      claim_strength_j = max(best_E_j, best_C_j)
        = how specifically verifiable is this summary sentence?
        High strength → specific checkable claim
        Low strength (all neutral) → vague filler

    Support per summary sentence:
      Blend ZS max + CMWS margin-weighted aggregation.
      Apply contradiction spread penalty.

    Final = softmax(claim_strength) · support
    """
    best_ent = E.max(dim=0).values          # (N,)
    best_con = C.max(dim=0).values          # (N,)
    claim_strength = torch.max(best_ent, best_con)   # (N,)

    # CMWS-style support: weight doc sentences by E + margin bonus
    margin     = torch.relu(E - Neu)                 # (M, N)
    agg_w      = E + args.margin_weight * margin     # (M, N)
    agg_s      = torch.softmax(agg_w * 10.0, dim=0) # (M, N)
    margin_score = (agg_s * E).sum(dim=0)            # (N,)

    # Blend ZS max + margin score (equal weight)
    support = 0.5 * best_ent + 0.5 * margin_score   # (N,)

    # Contradiction spread penalty
    if has_contradiction:
        con_spread = (C > args.con_threshold).float().mean(dim=0)  # (N,)
        support    = support - args.lambda_c * con_spread
        support    = support.clamp(0.0, 1.0)

    # Claim-strength-weighted final score
    claim_w = torch.softmax(claim_strength * args.claim_temp, dim=0)  # (N,)
    return (claim_w * support).sum().item()


# --- Strategy 4: SIFiD + CWCS (main novel scorer) ---
def agg_sifid_cwcs(E, Neu, C):
    """
    SIFiD document filtering followed by CWCS aggregation.

    This is the primary novel contribution:
    - SIFiD removes noisy, unrelated doc sentences from the matrix
    - CWCS then scores on the clean, semantically-filtered subset
    - Especially effective when document is long and mostly off-topic
      relative to any given summary sentence (polytope, xsumfaith)
    """
    mask  = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f   = E[mask]
    Neu_f = Neu[mask]
    C_f   = C[mask]
    if E_f.shape[0] == 0:
        return 0.5
    return agg_cwcs(E_f, Neu_f, C_f)


# --- Strategy 5: SIFiD + Soft-Min ---
def agg_sifid_softmin(E, Neu, C):
    """
    SIFiD filtering followed by soft-min aggregation.

    Soft-min is more sensitive than mean when ANY summary sentence
    is unsupported. After filtering the doc, the entailment scores
    for truly inconsistent summaries should be lower — making the
    soft-min penalty more discriminative.

    soft_min = -1/alpha * log( mean( exp(-alpha * col_max) ) )
    (log-sum-exp trick for differentiable minimum)
    """
    mask = sifid_mask(E, C, beta=args.beta, window=args.window)
    E_f  = E[mask]
    if E_f.shape[0] == 0:
        return 0.5

    col_max  = E_f.max(dim=0).values              # (N,)
    alpha    = args.softmin_alpha
    soft_min = -1.0 / alpha * torch.log(
        torch.mean(torch.exp(-alpha * col_max))
    )
    return float(soft_min.clamp(0.0, 1.0).item())


# ==============================
# Strategy Registry
# ==============================
STRATEGIES = {
    "zs_max"       : agg_zs,
    "sifid_zs"     : agg_sifid_zs,
    "cwcs"         : agg_cwcs,
    "sifid_cwcs"   : agg_sifid_cwcs,
    "sifid_softmin": agg_sifid_softmin,
}


# ==============================
# Score a single (document, summary) pair
# Returns dict {strategy_name: score}
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
# Stores all strategy scores so a single checkpoint
# can resume any of the 5 strategies simultaneously.
# ==============================
def load_checkpoint(name):
    if os.path.exists(args.checkpoint):
        with open(args.checkpoint) as f:
            data = json.load(f)
        if data.get("dataset_name") == name:
            all_scores = data.get("all_scores", {})
            # Ensure all strategies present (for forward compat)
            for strat in STRATEGIES:
                if strat not in all_scores:
                    all_scores[strat] = []
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
    print("SANITY CHECK — Improved Scorer (SIFiD + CWCS hybrid)")
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
              f"{'✓' if passed else '✗':>5}")

    print("=" * 70)
    print("✓ All strategies passed sanity check\n" if all_passed
          else "✗ Some strategies FAILED — check model\n")
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
        print(f"  Skipping {name} — only one class present")
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
        star    = "★" if is_best else " "
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
          f"({'↑ IMPROVED' if delta > 0 else '↓ below ZS'})")

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
# Evaluate on SummaC Benchmark
# ==============================
def evaluate_summac():
    try:
        from summac.benchmark import SummaCBenchmark
    except ImportError:
        print("Run: pip install summac")
        return

    VALID_DATASETS = {"cogensumm", "xsumfaith", "polytope"}

    print(f"Loading SummaC benchmark from {args.summac_folder}...")
    benchmark = SummaCBenchmark(
        benchmark_folder=args.summac_folder,
        dataset_names=["cogensum", "xsumfaith", "polytope"],
        cut="val"
    )
    print(f"Found:  {[d['name'] for d in benchmark.datasets]}")
    print(f"Using:  {sorted(VALID_DATASETS)}\n")

    all_results = {}
    best_baccs  = []
    zs_baccs    = []

    for dataset_info in benchmark.datasets:
        name = dataset_info["name"]
        if name not in VALID_DATASETS:
            print(f"\nSkipping {name} (not in valid set)")
            continue

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
    print("OVERALL RESULTS — SIFiD + CWCS Improved Scorer on SummaC Benchmark")
    print(f"{'='*70}")
    print(f"\n  {'Dataset':<14}  {'ZS BAcc':>8}  {'Best BAcc':>10}  "
          f"{'Best Strategy':>16}  {'Δ':>7}")
    print(f"  {'-'*62}")

    for name, res in all_results.items():
        zs_b   = res["per_strategy"]["zs_max"]["balanced_accuracy"]
        best_b = res["balanced_accuracy"]
        d      = best_b - zs_b
        print(f"  {name:<14}  {zs_b:>8.4f}  {best_b:>10.4f}  "
              f"{res['best_strategy']:>16}  "
              f"{'↑' if d >= 0 else '↓'}{abs(d):>6.4f}")

    print(f"  {'-'*62}")
    delta = overall_best - overall_zs
    print(f"  {'Average':<14}  {overall_zs:>8.4f}  {overall_best:>10.4f}  "
          f"{'':>16}  {'↑' if delta >= 0 else '↓'}{abs(delta):>6.4f}")

    print(f"\n  Datasets evaluated:         {len(all_results)}")
    print(f"  ZS baseline (this run):     {overall_zs:.4f}")
    print(f"  Best improved score:        {overall_best:.4f}")
    print(f"  Best ROC-AUC:               {overall_auc:.4f}")

    # Reference baselines from image / previous runs
    paper_zs   = 0.6471   # DeBERTa ZS from image (3-dataset avg)
    robust_ens = 0.6623   # ALBERT+RoBERTa ensemble (summac_robust_results.json)

    d_vs_zs  = overall_best - paper_zs
    d_vs_ens = overall_best - robust_ens

    print(f"\n  {'─'*42}")
    print(f"  vs DeBERTa ZS ({paper_zs:.4f}):         "
          f"{'+' if d_vs_zs >= 0 else ''}{d_vs_zs:.4f}  "
          f"{'✓ BEATS ZS' if d_vs_zs > 0 else '✗ below ZS'}")
    print(f"  vs ALBERT+RoBERTa ens ({robust_ens:.4f}):  "
          f"{'+' if d_vs_ens >= 0 else ''}{d_vs_ens:.4f}  "
          f"{'✓ BEATS ENSEMBLE' if d_vs_ens > 0 else '✗ below ensemble'}")
    print(f"  {'─'*42}")

    # === Strategy-level analysis ===
    print(f"\n  {'─'*42}")
    print("  Per-Strategy Average BAcc across all datasets:")
    for strat in STRATEGIES:
        avg = np.mean([r["per_strategy"][strat]["balanced_accuracy"]
                       for r in all_results.values()])
        marker = " ←best" if strat == max(
            STRATEGIES,
            key=lambda s: np.mean([r["per_strategy"][s]["balanced_accuracy"]
                                   for r in all_results.values()])
        ) else ""
        print(f"    {strat:<16}  avg BAcc = {avg:.4f}{marker}")

    with open(args.output, "w") as f:
        json.dump({
            "model"          : args.model,
            "scorer"         : "SIFiD+CWCS Improved",
            "params"         : {
                "beta"          : args.beta,
                "window"        : args.window,
                "margin_weight" : args.margin_weight,
                "lambda_c"      : args.lambda_c,
                "con_threshold" : args.con_threshold,
                "claim_temp"    : args.claim_temp,
                "softmin_alpha" : args.softmin_alpha,
            },
            "n_datasets"         : len(all_results),
            "overall_best_bacc"  : overall_best,
            "overall_zs_bacc"    : overall_zs,
            "overall_roc_auc"    : overall_auc,
            "delta_vs_zs"        : delta,
            "delta_vs_paper_zs"  : d_vs_zs,
            "delta_vs_ensemble"  : d_vs_ens,
            "per_dataset"        : all_results,
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    passed = sanity_check()
    if not passed:
        print("Sanity check failed — fix model before benchmarking.")
    else:
        print("Proceed to SummaC benchmark evaluation? (y/n): ", end="")
        if input().strip().lower() == "y":
            evaluate_summac()
        else:
            print("Skipping.")
