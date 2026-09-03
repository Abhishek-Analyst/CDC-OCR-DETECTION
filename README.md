# CDC-OCR-DETECTION
# Distorted Visual Sequence Pattern Recognition

CRNN (CNN + BiLSTM + CTC) for reading 6-character sequences from distorted grayscale images.

**ABHISHEK 22321003**

## Results

| Metric | Value |
|---|---|
| Validation CER | **0.0009** (0.09%) |
| Word Accuracy | **99.45%** |
| Wrong predictions | 11 / 2000 |

Measured on a held-out validation split (`random_state=42`, never trained on). The checkpoint was selected by minimising Val CER, so this figure is mildly optimistic as an estimate of test performance.

## Pipeline

```
image → OpenCV preprocessing → CNN → feature sequence → BiLSTM
      → per-frame char probabilities → CTC loss (train) / greedy decode (infer) → "X7K92B"
```

**OpenCV** — grayscale → median blur (3×3, kills salt-and-pepper noise) → CLAHE (clip 2.0, local contrast for uneven backgrounds). Runs at 0.10 ms/image, so it does not bottleneck the loader.

Binarisation is implemented but **off by default**. Otsu discards the anti-aliased grey pixels at stroke edges, which is exactly the signal the CNN relies on for low-contrast glyphs. Morphological opening is also off — it removes small *bright* structures, so it only helps if the noise is brighter than the glyphs.

**CNN** — 7 conv blocks (64→512 channels), BatchNorm + ReLU throughout.

Pooling is `(2,2), (2,2), –, (2,1), –, (2,1), –`. The later pools halve height but **preserve width**, giving 40 CTC timesteps instead of 10. With four stride-2 pools the sequence collapses to under two frames per character, which is far too tight for CTC to localise cleanly. An assert enforces `T ≥ 13`.

**BiLSTM** — 2 layers, hidden 256, bidirectional, dropout 0.25.

**CTC** — blank at index 0, `zero_infinity=True`, greedy decoding at inference.

## Training

| | |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | CosineAnnealingLR, `T_max = max_ep`, eta_min 1e-5 |
| Batch size | 64 |
| Epochs | 35, early stopping patience 8 |
| Precision | AMP on CUDA, no-op elsewhere |
| Augmentation | affine, perspective, colour jitter, random erasing |

Runtime ≈ 22 min on a Colab T4 at ~38 s/epoch.

## Notes on the training curve

Epochs 1–7 sit on a plateau: loss ~3.60, CER ~0.95. This is CTC **blank collapse** — the model emits only the blank token because that minimises loss before it has learned any character alignment. It breaks out around epoch 8 and drops quickly. Expected behaviour for CRNN + CTC from a cold start, not a bug.

Most of the final gain arrives in the last third of the schedule as the cosine LR anneals toward its floor. Interrupting early costs real accuracy: CER was 0.0222 at epoch 21 and 0.0009 by the end.

## Running it

Cells run top to bottom. Cell 4 auto-locates the dataset — it searches `/content`, `/kaggle/input`, cwd, Google Drive, and the usual local folders for a directory containing `train-labels.csv`, `train_images/` and `test_images/`. The folder name does not matter (`cig_ps`, `cig_ps 2`, a Downloads copy). If detection fails it prints what it actually found on disk; set `DATA_DIR_OVERRIDE` at the top of that cell.

On Colab it downloads and extracts automatically, and validates that the download is really a zip — gdown exits 0 even when Drive returns an HTML quota page.

Device selection probes each backend rather than trusting `is_available()`. The MPS branch specifically tests whether a CTC kernel exists, since several torch builds ship MPS without one.

Set `quick_run = True` in the config cell for a 3-epoch smoke test before committing to a full run.

## Implementation details worth knowing

**Labels are read with `dtype=str`.** Without it pandas turns `007123` into `7123`, which then fails the length-6 filter and silently drops valid training rows. Only matters if the label column is all-numeric, but it costs nothing.

**`persistent_workers=False` everywhere.** Interrupting the training cell kills the workers, and a stale loader then raises `DataLoader worker exited unexpectedly` on every subsequent cell until it is rebuilt. The saving at this scale is not worth the recovery hassle.

**Checkpoints are backed up before training.** A fresh run starts from `bst = inf` and overwrites `best_model.pth` on its first epoch, even though that epoch is far worse than a previous run's best. Any existing checkpoint is copied to `best_model_prev.pth` first.

**`cudnn.benchmark = True`** on CUDA — input size is fixed at 48×160, so autotuning pays off.

## Outputs

- `best_model.pth` — best checkpoint by Val CER (weights, epoch, CER)
- `best_model_prev.pth` — previous run's checkpoint, if one existed
- `submission_AryanMehra_23115025.csv` — 5000 rows, `image,prediction`

Submission validation checks row count against the test set, column names, filename uniqueness, and warns on empty predictions (decoder emitting only blanks).

## Limitations

- Greedy decoding only. Beam search with a character-level LM would likely help on the remaining ambiguous cases.
- Fixed 48×160 input; images are resized regardless of aspect ratio.
- The attention encoder-decoder comparison is not included in this notebook.
- No k-fold cross-validation — a single split, so the CER carries some variance.
- Test set has no labels, so test accuracy cannot be measured locally. Validation is the proxy.
