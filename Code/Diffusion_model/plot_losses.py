import argparse
import csv
import glob
import os

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "Save")
LOGS_DIR = os.path.join(SAVE_DIR, "lightning_logs")


def read_loss_csv(path, loss_key):
    rows = []
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append((int(row["Epoch"]), float(row[loss_key])))
    return rows


def clean_validation_rows(rows):
    if len(rows) < 2:
        return rows

    first_epoch, first_value = rows[0]
    second_epoch, second_value = rows[1]

    looks_like_sanity = (
        first_epoch == 1
        and second_epoch == 2
        and second_value > 0
        and first_value > 5 * second_value
    )

    if looks_like_sanity:
        return [(epoch - 1, value) for epoch, value in rows[1:]]

    return rows


def merge_series(paths, loss_key):
    merged = {}
    for path in sorted(paths):
        rows = read_loss_csv(path, loss_key)
        if loss_key == "Validation Loss":
            rows = clean_validation_rows(rows)
        for epoch, value in rows:
            merged[epoch] = value
    return sorted(merged.items())


def auto_discover(filename):
    pattern = os.path.join(LOGS_DIR, "version_*", filename)
    return sorted(glob.glob(pattern))


def plot_series(train_series, val_series, output_path=None, title="Diffusion Training Curves"):
    plt.figure(figsize=(9, 5))

    if train_series:
        plt.plot(
            [epoch for epoch, _ in train_series],
            [value for _, value in train_series],
            marker="o",
            linewidth=2,
            label="Train Loss",
        )

    if val_series:
        plt.plot(
            [epoch for epoch, _ in val_series],
            [value for _, value in val_series],
            marker="s",
            linewidth=2,
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot diffusion training and validation losses.")
    parser.add_argument("--train-csv", nargs="*", default=None, help="Optional list of training loss CSV files.")
    parser.add_argument("--val-csv", nargs="*", default=None, help="Optional list of validation loss CSV files.")
    parser.add_argument(
        "--output",
        default=os.path.join(SAVE_DIR, "loss_curves.png"),
        help="Output image path. Use an empty string to show interactively instead.",
    )
    parser.add_argument("--title", default="Diffusion Training Curves", help="Plot title.")
    args = parser.parse_args()

    train_paths = args.train_csv if args.train_csv else auto_discover("training_losses.csv")
    val_paths = args.val_csv if args.val_csv else auto_discover("validation_losses.csv")

    train_series = merge_series(train_paths, "Training Loss") if train_paths else []
    val_series = merge_series(val_paths, "Validation Loss") if val_paths else []

    if not train_series and not val_series:
        raise FileNotFoundError("No loss CSV files found. Run training first or pass --train-csv/--val-csv.")

    output_path = args.output if args.output else None
    plot_series(train_series, val_series, output_path=output_path, title=args.title)


if __name__ == "__main__":
    main()
