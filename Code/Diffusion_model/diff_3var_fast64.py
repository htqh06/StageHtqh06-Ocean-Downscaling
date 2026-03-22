import csv
import os

import diffusers
import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader

from Dataset_3var import Dataset_3var_train, Dataset_3var_valid


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Hyperparameters
l_files, n_files = 256, 37
data_path = os.path.join(PROJECT_ROOT, "data", "Copernicus_processed_data")
n_epochs = 40  # total epochs after resume
bsize = 8
n_ch = 128
resume_ckpt = os.path.join(
    SCRIPT_DIR,
    "Save",
    "lightning_logs",
    "version_2",
    "checkpoints",
    "epoch=19-step=2960.ckpt",
)


class SaveLossCallback(Callback):
    def __init__(self, save_dir, csv_file_train="training_losses.csv", csv_file_val="validation_losses.csv"):
        self.save_dir = save_dir
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append((trainer.current_epoch + 1, trainer.callback_metrics["train_loss"].item()))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append((trainer.current_epoch + 1, trainer.callback_metrics["val_loss"].item()))

    @staticmethod
    def _write_csv(path, header, rows):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            for row in rows:
                writer.writerow(row)

    def on_train_end(self, trainer, pl_module):
        run_dir = trainer.log_dir if trainer.log_dir else self.save_dir

        train_run_path = os.path.join(run_dir, self.csv_file_train)
        train_latest_path = os.path.join(self.save_dir, self.csv_file_train)
        self._write_csv(train_run_path, ["Epoch", "Training Loss"], self.train_losses)
        self._write_csv(train_latest_path, ["Epoch", "Training Loss"], self.train_losses)
        print(f"Training losses saved to {train_run_path}")

        val_run_path = os.path.join(run_dir, self.csv_file_val)
        val_latest_path = os.path.join(self.save_dir, self.csv_file_val)
        self._write_csv(val_run_path, ["Epoch", "Validation Loss"], self.val_losses)
        self._write_csv(val_latest_path, ["Epoch", "Validation Loss"], self.val_losses)
        print(f"Validation losses saved to {val_run_path}")


class DiffusionModel_3var(L.LightningModule):
    def __init__(self):
        super().__init__()

        # These buffers are only runtime scratch space and should not be saved/restored.
        self.register_buffer("noise_buffer", torch.empty(1, 3, 64, 64), persistent=False)
        self.register_buffer("steps_buffer", torch.empty(1, dtype=torch.long), persistent=False)

        # U-shaped residual network, with 3 channels for sss, sst and ssh
        self.model = diffusers.models.UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(n_ch, 2 * n_ch, 2 * n_ch, 4 * n_ch, 4 * n_ch),
            downsample_type="resnet",
            upsample_type="resnet",
        )

        # noise schedule using DDIM (faster than DDPM)
        self.scheduler = diffusers.schedulers.DDIMScheduler(
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False,
        )

    def generate_noise(self, shape, device):
        if self.noise_buffer.shape != shape:
            self.noise_buffer = torch.empty(*shape, device=device)
        return self.noise_buffer.normal_()

    def generate_steps(self, batch_size, device, max_step=1000):
        if self.steps_buffer.shape[0] != batch_size or self.steps_buffer.device != device:
            self.steps_buffer = torch.empty(batch_size, dtype=torch.long, device=device)
        return self.steps_buffer.random_(0, max_step)

    def training_step(self, images, batch_idx):
        images = images.to(memory_format=torch.channels_last)
        noise = self.generate_noise(images.shape, images.device)
        steps = self.generate_steps(images.size(0), images.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps).contiguous()
        residual = self.model(noisy_images, steps).sample.contiguous()

        loss = torch.nn.functional.l1_loss(residual, noise)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, images, batch_idx):
        images = images.to(memory_format=torch.channels_last)
        noise = self.generate_noise(images.shape, images.device)
        steps = self.generate_steps(images.size(0), images.device)
        noisy_images = self.scheduler.add_noise(images, noise, steps)
        residual = self.model(noisy_images, steps).sample
        val_loss = torch.nn.functional.mse_loss(residual, noise)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def load_state_dict(self, state_dict, strict=True):
        # Older checkpoints may contain these transient buffers with batch-shaped values.
        cleaned_state_dict = dict(state_dict)
        cleaned_state_dict.pop("noise_buffer", None)
        cleaned_state_dict.pop("steps_buffer", None)
        return super().load_state_dict(cleaned_state_dict, strict=strict)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    model = DiffusionModel_3var()

    dataset_train = Dataset_3var_train(l_files, n_files, data_path, "so_", "thetao_", "zos_")
    dataset_val = Dataset_3var_valid(data_path, "so_val.npy", "thetao_val.npy", "zos_val.npy")

    train_loader = DataLoader(dataset_train, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(dataset_val, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=True)

    checkpoint_dir = os.path.join(SCRIPT_DIR, "Save")
    save_loss_callback = SaveLossCallback(save_dir=checkpoint_dir)

    trainer = L.Trainer(
        accumulate_grad_batches=8,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        precision="16-mixed",
        max_epochs=n_epochs,
        num_sanity_val_steps=0,
        callbacks=[save_loss_callback],
        default_root_dir=checkpoint_dir,
    )

    fit_kwargs = {}
    if resume_ckpt and os.path.exists(resume_ckpt):
        fit_kwargs["ckpt_path"] = resume_ckpt
        print(f"Resuming training from checkpoint: {resume_ckpt}")
    else:
        print("Starting training from scratch.")

    trainer.fit(model, train_loader, valid_loader, **fit_kwargs)
