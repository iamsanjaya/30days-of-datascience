# Day 21 — CNNs: Transfer Learning (ResNet50 / Xception) + Grad-CAM

Part of [30days-of-datascience](https://github.com/iamsanjaya/30days-of-datascience) — Phase 3 (Deep Learning).

## Standard Task

Build a Cats vs Dogs classifier with a pretrained CNN backbone:

1. **Phase 1 — Frozen base.** Freeze all backbone layers, train only a small classification head.
2. **Phase 2 — Fine-tune.** Unfreeze the last 10 layers of the base, continue training at a much smaller learning rate.
3. Document the accuracy jump between Phase 1 and Phase 2.

The backbone is configurable (`config.BACKBONE`, either `"resnet50"` or `"xception"`) rather than hardcoded — see the Xception extension below.

## Out-of-Box Challenge — "What Does the Model Actually See?"

Implement Grad-CAM on the fine-tuned model. For 5 correctly classified and 5 misclassified test images, generate activation heatmaps showing where the model was looking. For the misclassified images specifically: was the model looking at the right region but still calling it wrong, or looking at the wrong region entirely? That judgment call — made by you, looking at `outputs/gradcam/<backbone>/gradcam_misclassified.png` — is the actual deliverable. The script only produces the evidence.

## Extension — Xception Comparison

Beyond the standard ResNet50 task, the pipeline also supports training an Xception backbone on the identical train/val/test split and comparing it head-to-head:

1. Set `config.BACKBONE = "resnet50"`, run `02` → `05`.
2. Set `config.BACKBONE = "xception"`, run `02` → `05` again (writes to separate model/output paths — nothing from the first run gets overwritten).
3. Run `06_compare_backbones.py` for a side-by-side metrics table and bar chart.

## Dataset

**Not included in this repo.** Using the **Microsoft Cats vs Dogs** dataset (the original Kaggle competition's data is gated behind a competition entry that wasn't accessible, so this is the freely downloadable mirror of the same underlying corpus):

> https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

Extract it so the structure lands at:

```
data/raw/PetImages/Cat/0.jpg
data/raw/PetImages/Cat/1.jpg
...
data/raw/PetImages/Dog/0.jpg
data/raw/PetImages/Dog/1.jpg
...
```

No synthetic data is used anywhere in this pipeline.

### Known data quality issue

This dataset ships with a small number of zero-byte / undecodable JPEGs — confirmed: `Cat/666.jpg`, `Dog/11702.jpg` (there may be others). Neither `tf.io.decode_jpeg` nor Keras's `ImageDataGenerator` skips these gracefully; both crash on read, the difference is only _when_ — `ImageDataGenerator` crashes mid-epoch once a bad file happens to get shuffled into a batch (losing whatever training already ran), while a naive `tf.data` pipeline would crash on first encounter too. `01_prepare_data.py` validates every file once, up front, by fully decoding it with PIL, and excludes anything that fails — so training never touches a bad file. The exclusion list is saved to `outputs/data_quality/corrupt_files_removed.csv`, not silently dropped.

## File Structure

```
day-21-cnn-transfer-learning/
├── config.py                  # all hyperparameters, paths, constants, backbone selection
├── 01_prepare_data.py         # scan raw dir, validate files, build stratified split
├── 02_train_frozen.py         # Phase 1: train head with frozen backbone
├── 03_finetune.py             # Phase 2: unfreeze last 10 layers, fine-tune
├── 04_compare_results.py      # evaluate both models on held-out test set
├── 05_gradcam_analysis.py     # Grad-CAM on correct vs misclassified samples
├── 06_compare_backbones.py    # ResNet50 vs Xception side-by-side (after running both)
├── utils/
│   ├── data.py                # scanning, validation, splitting, tf.data pipelines
│   ├── architecture.py        # backbone + head builder, layer unfreezing
│   ├── training.py            # compile/train/callbacks, history save/load
│   ├── gradcam.py             # Grad-CAM heatmap computation + overlay
│   └── visualization.py       # all plotting (curves, comparisons, grids)
├── data/raw/PetImages/        # <- place Kaggle dataset here (gitignored)
├── outputs/                   # generated at runtime (gitignored)
│   ├── splits/                # train.csv, val.csv, test.csv (shared across backbones)
│   ├── data_quality/          # corrupt_files_removed.csv
│   ├── curves/<backbone>/     # training curve PNGs, per backbone
│   ├── comparison/<backbone>/ # frozen vs fine-tuned metrics + plot, per backbone
│   ├── comparison/            # backbone_comparison.png (from 06)
│   └── gradcam/<backbone>/    # Grad-CAM grids + analysis log CSV, per backbone
├── models/                    # generated at runtime (gitignored)
│   ├── frozen_model_<backbone>.keras
│   └── finetuned_model_<backbone>.keras
└── README.md

learning-journal/
    └── day-21.md
```

## How to Run

```bash
# from project-env, with the dataset already extracted to data/raw/PetImages/
python 01_prepare_data.py      # one-time: validates files, builds the split
python 02_train_frozen.py      # Phase 1, for whichever config.BACKBONE is set
python 03_finetune.py          # Phase 2
python 04_compare_results.py   # frozen vs fine-tuned, this backbone
python 05_gradcam_analysis.py  # Grad-CAM, this backbone

# optional: repeat 02-05 with config.BACKBONE = "xception", then:
python 06_compare_backbones.py
```

`01_prepare_data.py` only needs to run once — the split is shared across backbones. `02`–`05` need to run once per backbone you want results for.

## Results

### Dataset Statistics

| Metric                 | Value  |
| ---------------------- | ------ |
| Original Images        | 25,000 |
| Valid Images           | 24,991 |
| Corrupt/Removed Images | 9      |
| Training Samples       | 17,493 |
| Validation Samples     | 3,749  |
| Test Samples           | 3,749  |

Corrupt and undecodable files were automatically detected using TensorFlow's native image decoder and excluded before training. The exclusion log is stored in:

```text
outputs/data_quality/corrupt_files_removed.csv
```

---

### ResNet50 Results

#### Frozen Backbone

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 98.85% |
| Precision | 99.25% |
| Recall    | 98.45% |
| F1 Score  | 98.85% |
| ROC-AUC   | 99.93% |

#### Fine-Tuned Backbone

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.01% |
| Precision | 99.30% |
| Recall    | 98.72% |
| F1 Score  | 99.01% |
| ROC-AUC   | 99.95% |

---

### Xception Results

#### Frozen Backbone

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.44% |
| Precision | 99.63% |
| Recall    | 99.25% |
| F1 Score  | 99.44% |
| ROC-AUC   | 99.95% |

#### Fine-Tuned Backbone

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 99.47% |
| Precision | 99.52% |
| Recall    | 99.41% |
| F1 Score  | 99.47% |
| ROC-AUC   | 99.96% |

---

### Backbone Comparison (Fine-Tuned Models)

| Metric    | ResNet50 | Xception | Delta  |
| --------- | -------- | -------- | ------ |
| Accuracy  | 99.01%   | 99.47%   | +0.45% |
| Precision | 99.30%   | 99.52%   | +0.22% |
| Recall    | 98.72%   | 99.41%   | +0.69% |
| F1 Score  | 99.01%   | 99.47%   | +0.46% |
| ROC-AUC   | 99.95%   | 99.96%   | +0.01% |

### Best Model

**Xception** achieved the best overall performance on the held-out test set:

- Accuracy: **99.47%**
- Precision: **99.52%**
- Recall: **99.41%**
- F1 Score: **99.47%**
- ROC-AUC: **99.96%**

While ResNet50 trained and evaluated faster, Xception consistently outperformed it across all evaluation metrics.

---

### Grad-CAM Analysis

Grad-CAM was applied to both correctly classified and misclassified test samples.

The generated heatmaps showed that the model generally focused on semantically meaningful regions such as:

- Animal faces
- Eyes
- Ears
- Fur patterns
- Body contours

Most errors occurred on visually ambiguous images where cat and dog features overlapped significantly, rather than from attention being concentrated on irrelevant background regions.

Generated outputs:

```text
outputs/gradcam/<backbone>/gradcam_correct.png
outputs/gradcam/<backbone>/gradcam_misclassified.png
```

---

### Key Takeaways

- Transfer learning with ImageNet-pretrained CNNs is highly effective for small and medium-sized image classification tasks.
- Fine-tuning provided modest but measurable gains after frozen feature extraction.
- Xception consistently outperformed ResNet50 on the same dataset split.
- Grad-CAM provides an interpretable mechanism for understanding model decisions.
- Data quality validation is critical because corrupted images can silently break training pipelines.

## Design Notes

- **No copied image directories.** `utils/data.py` scans the existing `Cat/`/`Dog/` folders into a (filepath, label) DataFrame and drives `tf.data.Dataset.from_tensor_slices` off that, rather than physically reorganizing 25k files. Splits are saved as CSVs so every script works from an identical, reproducible, already-validated split.
- **Backbone-aware preprocessing.** ResNet50 expects caffe-style (BGR, mean-subtracted) input; Xception expects images scaled to `[-1, 1]`. Mixing these up doesn't error — it just silently tanks accuracy — so the dispatch lives in one place (`data._preprocess_input`) rather than being duplicated per script.
- **BatchNorm handling during fine-tuning.** When unfreezing the last 10 layers in `architecture.unfreeze_top_layers()`, any BatchNorm layers within that range are explicitly kept frozen (inference mode). Retraining BatchNorm statistics on a small fine-tuning batch size is a well-known source of instability.
- **`tf.keras.optimizers.legacy.Adam`** is used in `training.compile_model()` — consistent with the Day 19/20 fix for `tensorflow-metal` graph remapper crashes on BatchNorm-heavy models.
- **Grad-CAM hooks the last conv block** of whichever backbone is active (`conv5_block3_out` for ResNet50, `block14_sepconv2_act` for Xception) — both confirmed against `base_model.summary()`, not assumed.
- Subsampling is supported via `config.SAMPLE_SIZE` (set to an int instead of `None`) if a full run is too slow for fast iteration on the M4 — stratified, so class balance is preserved at any sample size.

## Known Gaps / TODO

- `requirements.txt` does not yet need new packages for this day (TensorFlow/Keras, scikit-learn, Pillow all already present from prior days) — no action needed before Day 21.
- Days 23 NLP/BERT work will still need `transformers`, `torch`, `gensim` added before that day arrives — unrelated to this folder, noted here only as a standing reminder.
