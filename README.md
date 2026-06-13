# Summary Inconsistency Detector

Detecting factual inconsistencies in generated summaries using Transformer-based Natural Language Inference (NLI).

## Overview

Modern summarization models often generate fluent summaries that contain subtle factual errors or contradictions with respect to the source document. This project addresses that problem by framing summary consistency verification as a Natural Language Inference task.

The system evaluates whether a summary is factually supported by its source document and produces a consistency score based on entailment and contradiction signals from a fine-tuned DeBERTa-v3 model.

## Key Contribution

Traditional SummaC-ZS scoring relies only on entailment probabilities.

This project introduces **Contradiction-Gated Entailment (CGE)**:

S(i,j) = E(i,j) × (1 − C(i,j))

where:

* E(i,j) = entailment probability
* C(i,j) = contradiction probability

This formulation suppresses uncertain predictions where both entailment and contradiction receive high confidence and improves factual consistency detection.

## Architecture

Source Document
↓
Sentence Segmentation
↓
NLI Matrix Construction
↓
DeBERTa-v3 NLI Model
↓
Entailment / Neutral / Contradiction Scores
↓
Contradiction-Gated Entailment
↓
Document Consistency Score

## Training Pipeline

### Stage 1: NLI Training

Fine-tune DeBERTa-v3-base on:

* SNLI
* MNLI

### Stage 2: Contradiction Sensitivity Enhancement

Further fine-tune on:

* VitaminC

with contradiction-aware loss weighting.

### Stage 3: Probability Calibration

Apply temperature scaling to improve probability calibration and reduce overconfidence.

## Datasets

### Training

* SNLI
* MNLI
* VitaminC

### Evaluation

* CoGenSumm
* XSumFaith
* PolyTope
* FRANK

## Results

| Dataset   | Baseline | CGE   |
| --------- | -------- | ----- |
| CoGenSumm | 70.44    | 70.91 |
| XSumFaith | 66.19    | 66.31 |
| PolyTope  | 57.46    | 57.88 |
| FRANK     | 81.46    | 81.22 |

Average Balanced Accuracy (Primary Datasets)

* Baseline: 64.70%
* Contradiction-Gated Entailment: 67.25%

Improvement: +2.55 percentage points

## Features

* Transformer-based NLI
* Document-level consistency scoring
* Contradiction-aware aggregation
* Temperature calibration
* Benchmark evaluation
* Multiple aggregation strategy comparison

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* DeBERTa-v3
* NLTK
* NumPy
* Pandas
* Scikit-Learn

## Repository Structure

```text
train_large.py
vitaminc_finetune_large.py
calibrate_large.py

models/
datasets/
results/
configs/

evaluate.py
score.py
```

## Future Work

* Learnable contradiction gates
* Larger DeBERTa backbones
* Long-context NLI models
* FactCC and SummEval evaluation
* Retrieval-augmented consistency verification

