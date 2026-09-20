# Project context

This project studies **semi-supervised cervical vertebra segmentation in videofluoroscopy (VFSS)**.

**Paper target:** CMPB (Computer Methods and Programs in Biomedicine). CBM as backup.

**Tentative title:** Semi-Supervised Learning for Cervical Vertebra Segmentation in Videofluoroscopy: A Systematic Evaluation of Pool Construction Strategies

**Core contribution:** Empirical study showing that (1) SSL consistently improves over supervised training for vertebra segmentation in VFSS within a fixed pipeline, (2) unlabeled pool size strongly influences SSL performance with diminishing returns beyond an intermediate range, (3) selection policy has a smaller, method-dependent effect, (4) a strong self-configuring baseline (nnU-Net) achieves higher absolute performance through its optimized pipeline but saturates rapidly across label fractions.

This project is **NOT** a novel SSL algorithm. It is an empirical/methodological study of unlabeled-pool construction inside a standard teacher-student framework, benchmarked against strong supervised references.

---

## Current empirical status (CONFIRMED)

> **Canonical result numbers live in `tesis_seg/RESULTS_FINAL.md`** (verified against
> the official `runs_final_v1/` and `runs_inca_final_v1/` directories, correct seed
> counts: 5 seeds for architectures / r10 / UNM fractions, 3 seeds otherwise).
> The numeric tables that used to be here were from the OLD `runs/` set and were
> ~2 points off (e.g. supervised .821→.801, PL r3 .838→.852, MT all-lateral .847→.860).
> They were removed to avoid reusing wrong figures — do NOT re-add numbers here; cite
> RESULTS_FINAL.md instead. Per-vertebra boundary diagnostic: `resultados/diagnostico_dice/`
> (method B) and `tesis_seg/analysis/README.md`.

Qualitative summary (see RESULTS_FINAL.md for exact values):
- SSL improves over the supervised U-Net++ baseline on UNM; best pools are intermediate-to-large
  (MT random r10, MT/PL all-lateral). Selection policy (temporal vs random) has a small,
  inconsistent effect.
- On INCA at 100% labels the supervised baseline is already saturated → SSL is flat. On the
  low-label INCA patient10 subset SSL helps clearly (MT r10 best).
- nnU-Net (self-configuring, 1024×1024, single-run reference) reaches higher absolute F1 but
  saturates at 25% labels. It is NOT a controlled comparator (differs in resolution,
  augmentation, normalization, epochs, architecture depth).

### 1024×1024 experiments — ABANDONED

Training at 1024×1024 with batch_size=1 (GPU memory limit) led to training instability and poor generalization. Supervised 1024 seed 0 = .750 (vs .821 at 320). SSL 1024 seed 0 = .786. Qualitative inspection showed recurring undersegmented and fragmented masks. The issue is insufficient regularization with effective batch size of 6, not a code bug. Reported in Discussion as evidence that nnU-Net's advantage stems from its complete training regime, not resolution alone.

### nnU-Net + SSL self-training — IN PROGRESS

Cells F-J added to notebook 02. Using nnU-Net 25% and 100% as teachers to generate pseudo-labels on unlabeled pool r10 (3,937 frames), then retraining nnU-Net with real labels + pseudo-labels. Goal: demonstrate SSL helps even a strong supervised baseline.

**Confirmed findings:**
- SSL improves over the supervised U-Net++ baseline within our pipeline (exact deltas in RESULTS_FINAL.md)
- Pool size strongly influences performance: optimal range r=10–15 (~4,000–5,600 frames)
- Using all lateral frames (74,774) does not improve over r=10 (3,937 frames)
- Selection policy (temporal vs random) has smaller, inconsistent effect
- nnU-Net saturates at 25% labels (.896 vs .908 at 100%)
- Kim BiFPN-UNet without pretraining underperforms pretrained generic architectures

---

## Repo structure

```
workspace root (this directory):
    CLAUDE.md                          # This file
    README.md                          # Project README
    prep_label_fractions.py            # Generates nested label subsets
    prep_unm_unlabeled_new.py          # Creates temporal unlabeled pools
    prep_unm_unlabeled_std_clean.py    # Creates matched random control pools (AP-filtered)
    plot_perseed_f1.py                 # Per-seed F1 figure
    refresh_run_reports.py             # Regenerates run_report.json from artifacts
    summarize_runs.py                  # Consolidates boundary metrics
    paper_figures/                     # Figure generation pipeline for paper
    label_fractions/                   # Nested label subsets (frac_25, frac_50, frac_75)
    runs/                              # All experiment outputs (seed-specific dirs)
    train/, val/, test/                # Image and mask directories
    unlabeling_r3_max0/               # Temporal pool r=3 (1,257 frames)
    unlabeling_r5_max0/               # Temporal pool r=5 (2,066 frames)
    unlabeling_r7_max0/               # Temporal pool r=7 (2,849 frames)
    unlabeling_r10_max0/              # Temporal pool r=10 (3,937 frames)
    unlabeling_r15_max0/              # Temporal pool r=15 (5,590 frames)
    unlabeling_r20_max0/              # Temporal pool r=20 (7,081 frames)
    unlabeling_all_lateral/           # All lateral frames (74,774 frames)
    unlabeling_std_matched_r3/        # Random matched pool r=3
    unlabeling_std_matched_r5/        # Random matched pool r=5
    unlabeling_std_matched_r7/        # Random matched pool r=7
    unlabeling_std_matched_r10/       # Random matched pool r=10
    unlabeling_std_matched_r15/       # Random matched pool r=15
    unlabeling_std_matched_r20/       # Random matched pool r=20

    tesis_seg/                         # Training code repo
        notebooks/
            01_train_eval_colab.ipynb   # Main training notebook (96+ runs)
            02_nnunet_baseline.ipynb    # nnU-Net baseline + label efficiency + SSL self-training
        src/
            __init__.py
            defaults.py                # DEFAULT_CONFIG
            preprocessing.py           # Image preprocessing
            augmentations.py           # Supervised + weak/strong augmentations
            datasets.py                # SegmentationDataset, UnlabeledFramesDataset
            models.py                  # Model factory: unet, unetpp, fpn, deeplabv3+, transunet, bifpn_unet
            bifpn_unet.py              # Kim-inspired BiFPN-U-Net(T) implementation
            losses.py                  # BCE + Dice
            metrics.py                 # F1, IoU, BF1, ASSD, HD95
            train.py                   # Training loop (supervised + SSL)
            evaluate.py                # Evaluation, export predictions, run_report.json
            visualization.py           # Prediction visualization
            ruler_eval.py              # C2-C4 landmark comparison
            sanity_checks.py           # Pre-training sanity checks
            utils.py                   # Seed, EMA, JSON utilities
            transunet/                 # TransUNet implementation
                __init__.py
                vit_seg_configs.py
                vit_seg_modeling.py
                vit_seg_modeling_resnet_skip.py
        README.md
        requirements.txt
```

---

## Dataset

- **Source:** UNM TalkBank Dysphagia database (35 videos)
- **Split:** 23 train / 5 val / 7 test videos (no patient overlap)
- **Labeled frames:** 218 train, 44 val, 63 test (total 325)
- **Annotation:** 10 uniformly sampled frames per video, by trained annotators
- **Masks:** Binary, cervical vertebral region C2–C4
- **Resolution:** ~934×962 original, resized to 320×320 (grayscale, zero-padded, [0,1])
- **AP filtering:** 17 of 23 training videos have AP projection change; frames after cutoff excluded from all unlabeled pools

---

## SSL methods implemented

### Pseudo-labeling (ssl_method="pseudo_label", DEFAULT)
- Teacher = EMA of student (decay=0.99)
- Hard pseudo-labels at threshold 0.5
- Confidence mask at tau=0.95
- Loss: BCE-with-logits, masked by confidence

### Mean Teacher (ssl_method="mean_teacher")
- Same teacher EMA, same augmentations
- Loss: MSE between student and teacher probabilities
- No threshold, no confidence mask, all pixels contribute

Both methods share: EMA update, lambda_u=0.05 ramp, semi_warmup_epochs=20, weak/strong augmentations.
`semi_start_epoch` is dataset-specific: **15 for UNM, 7 for INCA** (verified across the 91 SSL runs
in `runs_final_v1/` and `runs_inca_final_v1/`). The value 30 appears only in supervised configs,
where `use_semi: False` makes it inert.

---

## Experiments in runs/

### Completed — SSL (84 runs: 7 radii × 2 policies × 2 methods × 3 seeds)
- `semi_r{3,5,7,10,15,20}/seed_{0,1,2}` — PL temporal
- `semi_std_matched_r{3,5,7,10,15,20}/seed_{0,1,2}` — PL random
- `semi_all_lateral/seed_{0,1,2}` — PL all lateral frames
- `mean_teacher_r{3,5,7,10,15,20}/seed_{0,1,2}` — MT temporal
- `mean_teacher_std_matched_r{3,5,7,10,15,20}/seed_{0,1,2}` — MT random
- `mean_teacher_all_lateral/seed_{0,1,2}` — MT all lateral frames

### Completed — Supervised baselines
- `supervised/seed_{0,1,2}` — U-Net++ baseline (218 labels)
- `supervised_unet/seed_{0,1,2}` — U-Net baseline
- `supervised_deeplabv3plus/seed_0` — DeepLabV3+
- `supervised_fpn/seed_0` — FPN
- `supervised_transunet/seed_0` — TransUNet
- `supervised_bifpn_unet/seed_{0,1,2}` — Kim BiFPN-UNet(T)

### Completed — Label efficiency (3 seeds each)
- `supervised_frac{25,50,75}/seed_{0,1,2}`
- `semi_r10_frac{25,50,75}/seed_{0,1,2}`

### Abandoned — 1024×1024 experiments
- `supervised_1024/seed_{0,1,2}` — poor results due to batch size constraints
- `semi_r10_1024/seed_0` — same issue

### In progress — nnU-Net SSL self-training
- Cells F-J in notebook 02_nnunet_baseline.ipynb
- Dataset502_VFSS_SSL25 (25% teacher + pseudo-labels)
- Dataset503_VFSS_SSL100 (100% teacher + pseudo-labels)

---

## Pending tasks

1. nnU-Net SSL self-training results (Cells F-J)
2. Inter-rater agreement (15-20 frames)
3. Paper v14 → final version with all results
4. Bootstrap/CIs for main comparisons (optional)
5. Error analysis figures (optional)

---

## Key config values (all main experiments)

```python
"arch": "unetpp"
"backbone": "efficientnet-b3"
"image_preproc": "base"
"mask_smoothing": "none"
"target_size": (320, 320)
"lr": 0.001
"batch_size": 5
"num_augmented": 5
"epochs": 2000
"patience_es": 40
"lambda_u": 0.05
"tau": 0.95
"ema_decay": 0.99
"semi_start_epoch": 15   # UNM; INCA uses 7
"semi_warmup_epochs": 20
"use_temp_consistency": False
"lambda_t": 0.0
```

### 1024×1024 experiments used different values (abandoned):
```python
"target_size": (1024, 1024)
"batch_size": 1
"lr": 0.0003
"patience_es": 50
```

---

## Design decisions already made

- **Architecture:** U-Net++ selected from 5-architecture comparison for low variance
- **Preprocessing:** "base" (no CLAHE). CLAHE tested and worsened performance
- **Radii:** 7 values from r=3 to r=20 plus all_lateral, covering ~100ms to ~667ms at 30fps
- **AP filtering:** Random pools exclude frames after AP projection change
- **Kim baseline:** Adaptation of Kim 2021 BiFPN-U-Net(T), not exact reproduction
- **nnU-Net:** Reported as contextual reference, not controlled comparator
- **1024 experiments:** Abandoned due to training instability with reduced batch size
- **Temporal consistency loss:** Implemented but disabled (use_temp_consistency=False)

---

## Naming conventions

| Internal name | Paper name |
|---|---|
| supervised | Supervised |
| semi_r{N} | PL Temporal-r{N} |
| semi_std_matched_r{N} | PL Random-r{N} |
| semi_all_lateral | PL All-lateral |
| mean_teacher_r{N} | MT Temporal-r{N} |
| mean_teacher_std_matched_r{N} | MT Random-r{N} |
| mean_teacher_all_lateral | MT All-lateral |
| supervised_bifpn_unet | Kim BiFPN-UNet(T) |

---

## Working rules for Claude

- Do NOT overclaim novelty — this is an empirical study, not a new method
- Do NOT modify existing experiment results or saved artifacts
- Do NOT mix outputs across seeds
- Do NOT confuse temporal pool selection with temporal consistency loss
- Prefer minimal, explicit code changes
- Before implementing, summarize the proposed changes
- Each run must use seed-specific directories: runs/{exp_name}/seed_{N}/
- All existing pseudo-labeling behavior must remain bit-for-bit identical
- nnU-Net is a contextual reference, NOT a controlled comparator
- Kim is an adaptation, NOT an exact reproduction
- Use "in our setting" / "appears to" language, not universal claims
