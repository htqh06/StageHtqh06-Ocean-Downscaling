import csv
import os
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from guided_sampling_3var import guidance_3var
from obs_operator_3var import downsample_to_mean


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "Copernicus_processed_data"
SAVE_DIR = SCRIPT_DIR / "Save"

CHECKPOINTS = {
    "epoch19": SCRIPT_DIR / "Save" / "lightning_logs" / "version_2" / "checkpoints" / "epoch=19-step=2960.ckpt",
    "epoch39": SCRIPT_DIR / "Save" / "lightning_logs" / "version_4" / "checkpoints" / "epoch=39-step=5920.ckpt",
}

TEST_FILES = {
    "sss": DATA_DIR / "so_test.npy",
    "sst": DATA_DIR / "thetao_test.npy",
    "ssh": DATA_DIR / "zos_test.npy",
}

DEFAULT_INDICES = [42, 183, 320]
DEFAULT_CROP = (30, 33)
DEFAULT_TIMESTEPS = 50
DEFAULT_BASE_SEED = 526557
TEST_YEAR = 2020


def compute_metrics(pred, true_sss, true_sst, true_ssh, obs_sss):
    pred_sss = pred[0]
    pred_sst = pred[1]
    pred_ssh = pred[2]
    pred_sss_lr = downsample_to_mean(torch.tensor(pred_sss, dtype=torch.float32)).cpu().numpy()

    return {
        "sss_mse": float(np.mean((pred_sss - true_sss) ** 2)),
        "sst_mse": float(np.mean((pred_sst - true_sst) ** 2)),
        "ssh_mse": float(np.mean((pred_ssh - true_ssh) ** 2)),
        "sss_lr_mse": float(np.mean((pred_sss_lr - obs_sss) ** 2)),
        "overall_mse": float(
            np.mean(
                np.stack(
                    [
                        (pred_sss - true_sss) ** 2,
                        (pred_sst - true_sst) ** 2,
                        (pred_ssh - true_ssh) ** 2,
                    ],
                    axis=0,
                )
            )
        ),
    }


def test_index_to_date(index, year=TEST_YEAR):
    return date(year, 1, 1) + timedelta(days=index)


def save_metrics_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "checkpoint",
                "index",
                "seed",
                "r1",
                "r2",
                "sss_mse",
                "sst_mse",
                "ssh_mse",
                "sss_lr_mse",
                "overall_mse",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_visual_comparison(images_by_checkpoint, true_images, output_path, sample_index, sample_date):
    channel_names = ["SSS", "SST", "SSH"]
    columns = ["Truth"] + list(images_by_checkpoint.keys())
    figure, axes = plt.subplots(3, len(columns), figsize=(4.2 * len(columns), 10))

    for row, channel_name in enumerate(channel_names):
        truth = true_images[row]
        truth_vmin = float(truth.min())
        truth_vmax = float(truth.max())

        ax = axes[row, 0]
        ax.imshow(truth, cmap="viridis", vmin=truth_vmin, vmax=truth_vmax)
        ax.set_title(f"{channel_name} Truth")
        ax.axis("off")

        for col, checkpoint_name in enumerate(images_by_checkpoint.keys(), start=1):
            pred = images_by_checkpoint[checkpoint_name][row]
            ax = axes[row, col]
            ax.imshow(pred, cmap="viridis", vmin=truth_vmin, vmax=truth_vmax)
            ax.set_title(f"{channel_name} {checkpoint_name}")
            ax.axis("off")

    figure.suptitle(
        f"Guided Sampling Comparison on Test Index {sample_index} ({sample_date.isoformat()})",
        fontsize=16,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    r1, r2 = DEFAULT_CROP
    rows = []
    outputs_for_first_index = {}
    true_images_for_first_index = None

    for index in DEFAULT_INDICES:
        seed = DEFAULT_BASE_SEED + index
        for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
            final_image, true_sss, true_sst, true_ssh, obs_sss_clean, _ = guidance_3var(
                checkpoint=str(checkpoint_path),
                fname_sss=str(TEST_FILES["sss"]),
                fname_sst=str(TEST_FILES["sst"]),
                fname_ssh=str(TEST_FILES["ssh"]),
                index=index,
                r1=r1,
                r2=r2,
                num_timesteps=DEFAULT_TIMESTEPS,
                seed=seed,
                device=torch.device("cuda:0"),
            )

            metrics = compute_metrics(final_image, true_sss, true_sst, true_ssh, obs_sss_clean)
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "index": index,
                    "seed": seed,
                    "r1": r1,
                    "r2": r2,
                    **metrics,
                }
            )

            if index == DEFAULT_INDICES[0]:
                outputs_for_first_index[checkpoint_name] = final_image
                true_images_for_first_index = [true_sss, true_sst, true_ssh]

    sample_date = test_index_to_date(DEFAULT_INDICES[0])
    metrics_path = SAVE_DIR / "checkpoint_comparison_metrics.csv"
    figure_path = SAVE_DIR / f"checkpoint_comparison_{sample_date.isoformat()}.png"
    save_metrics_csv(rows, metrics_path)

    if true_images_for_first_index is not None:
        save_visual_comparison(
            outputs_for_first_index,
            true_images_for_first_index,
            figure_path,
            sample_index=DEFAULT_INDICES[0],
            sample_date=sample_date,
        )

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved figure to {figure_path}")

    for checkpoint_name in CHECKPOINTS:
        checkpoint_rows = [row for row in rows if row["checkpoint"] == checkpoint_name]
        mean_overall = np.mean([row["overall_mse"] for row in checkpoint_rows])
        mean_sss = np.mean([row["sss_mse"] for row in checkpoint_rows])
        mean_sst = np.mean([row["sst_mse"] for row in checkpoint_rows])
        mean_ssh = np.mean([row["ssh_mse"] for row in checkpoint_rows])
        mean_sss_lr = np.mean([row["sss_lr_mse"] for row in checkpoint_rows])
        print(
            checkpoint_name,
            {
                "mean_overall_mse": float(mean_overall),
                "mean_sss_mse": float(mean_sss),
                "mean_sst_mse": float(mean_sst),
                "mean_ssh_mse": float(mean_ssh),
                "mean_sss_lr_mse": float(mean_sss_lr),
            },
        )


if __name__ == "__main__":
    main()
