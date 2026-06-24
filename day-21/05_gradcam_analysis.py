"""
Day 21 — Step 5: Out-of-Box Challenge — "What Does ResNet Actually See?"

Generates Grad-CAM heatmaps for 5 correctly classified and 5
misclassified test images using the fine-tuned model. For each
misclassified image, the saved CSV + grid lets you judge by eye: was
the model looking at the right region and still wrong, or looking at
the wrong region entirely? That written judgment is the actual
deliverable here — this script only produces the evidence.

Run:
    python 05_gradcam_analysis.py
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import config
from utils import data, gradcam, visualization


def load_raw_image(filepath: str) -> np.ndarray:
    """Load an image as uint8 RGB at IMG_SIZE, independent of the
    ResNet50-preprocessed tensor used for prediction — this is purely
    for human-viewable overlays."""
    img = Image.open(filepath).convert("RGB").resize(config.IMG_SIZE)
    return np.array(img)


def main() -> None:
    print(f"Backbone: {config.BACKBONE}")
    _, _, test_df = data.load_splits()
    test_ds = data.build_dataset(test_df, augment=False, shuffle=False)

    print(f"Loading fine-tuned model from {config.FINETUNED_MODEL_PATH}...")
    model = tf.keras.models.load_model(config.FINETUNED_MODEL_PATH)

    print("Generating predictions on test set...")
    y_prob = model.predict(test_ds).flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_df["label"].values

    test_df = test_df.copy()
    test_df["pred_label"] = y_pred
    test_df["confidence"] = y_prob
    test_df["correct"] = y_pred == y_true

    correct_pool = test_df[test_df["correct"]].sample(
        n=min(config.NUM_CORRECT_SAMPLES, test_df["correct"].sum()),
        random_state=config.SEED,
    )
    misclassified_pool = test_df[~test_df["correct"]]
    n_misclassified = min(config.NUM_MISCLASSIFIED_SAMPLES, len(misclassified_pool))
    if n_misclassified == 0:
        print(
            "WARNING: zero misclassified test samples — model is perfect on this "
            "test set (or test set is too small). Skipping misclassified grid."
        )
    misclassified_pool = misclassified_pool.sample(
        n=n_misclassified, random_state=config.SEED
    )

    label_names = {0: "cat", 1: "dog"}

    def build_samples(pool: pd.DataFrame) -> list[dict]:
        samples = []
        for _, row in pool.iterrows():
            raw_img = load_raw_image(row["filepath"])
            img_tensor, _ = data.load_and_preprocess_single(
                row["filepath"], row["label"]
            )
            img_tensor = tf.expand_dims(img_tensor, axis=0)

            heatmap = gradcam.make_gradcam_heatmap(img_tensor, model)
            overlay = gradcam.overlay_heatmap(raw_img, heatmap)

            samples.append(
                {
                    "filepath": row["filepath"],
                    "original": raw_img,
                    "overlay": overlay,
                    "true_label": label_names[int(row["label"])],
                    "pred_label": label_names[int(row["pred_label"])],
                    "confidence": float(row["confidence"]),
                }
            )
        return samples

    print("Generating Grad-CAM heatmaps for correctly classified samples...")
    correct_samples = build_samples(correct_pool)
    visualization.plot_gradcam_grid(
        correct_samples,
        save_path=config.GRADCAM_DIR / "gradcam_correct.png",
        title="Grad-CAM — Correctly Classified Samples",
    )

    if n_misclassified > 0:
        print("Generating Grad-CAM heatmaps for misclassified samples...")
        misclassified_samples = build_samples(misclassified_pool)
        visualization.plot_gradcam_grid(
            misclassified_samples,
            save_path=config.GRADCAM_DIR / "gradcam_misclassified.png",
            title="Grad-CAM — Misclassified Samples",
        )
    else:
        misclassified_samples = []

    # Forensic log for the written analysis (root-cause reasoning is
    # yours to fill in — this just gives you the evidence table).
    log_rows = []
    for s in correct_samples + misclassified_samples:
        log_rows.append(
            {
                "filepath": s["filepath"],
                "true_label": s["true_label"],
                "pred_label": s["pred_label"],
                "confidence": s["confidence"],
                "correct": s["true_label"] == s["pred_label"],
            }
        )
    log_df = pd.DataFrame(log_rows)
    config.GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.GRADCAM_DIR / "gradcam_analysis_log.csv"
    log_df.to_csv(log_path, index=False)

    print(f"\nGrad-CAM grids and analysis log saved to {config.GRADCAM_DIR}")
    print("Open gradcam_misclassified.png and judge: is the model looking at the")
    print("right region and still wrong, or looking at the wrong region entirely?")


if __name__ == "__main__":
    main()
