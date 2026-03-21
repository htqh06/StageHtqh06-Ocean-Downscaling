import torch
from torch.utils.data import DataLoader
import diffusers
import lightning as L
from Dataset_3var import Dataset_3var_train, Dataset_3var_valid  
import os
from lightning.pytorch.callbacks import Callback
import csv  

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

# Hyperparameters
l_files, n_files = 256, 37
data_path = "f:/StageTH/data/Copernicus_processed_data" #path to the numpy files containing the data
n_epochs = 1
bsize = 8
n_ch = 128

# Callback to save losses
class SaveLossCallback(Callback):
    def __init__(self, save_dir, csv_file_train="training_losses.csv", csv_file_val="validation_losses.csv"):
        self.csv_file_train = os.path.join(save_dir, csv_file_train)
        self.csv_file_val = os.path.join(save_dir, csv_file_val)
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss" in trainer.callback_metrics:
            self.train_losses.append(trainer.callback_metrics["train_loss"].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss" in trainer.callback_metrics:
            self.val_losses.append(trainer.callback_metrics["val_loss"].item())

    def on_train_end(self, trainer, pl_module):
        with open(self.csv_file_train, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Training Loss"])
            for epoch, loss in enumerate(self.train_losses, start=1):
                writer.writerow([epoch, loss])
        print(f"Training losses saved to {self.csv_file_train}")

        with open(self.csv_file_val, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Validation Loss"])
            for epoch, loss in enumerate(self.val_losses, start=1):
                writer.writerow([epoch, loss])
        print(f"Validation losses saved to {self.csv_file_val}")

# the core class of the diffusion model, using the diffusers library (see huggingface documentation)
class DiffusionModel_3var(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.register_buffer("noise_buffer", torch.empty(1, 3, 64, 64))  
        self.register_buffer("steps_buffer", torch.empty(1, dtype=torch.long))
        
        # U-shaped residual network, with 3 channels for sss, sst and ssh
        self.model = diffusers.models.UNet2DModel(
            sample_size=64,
            in_channels=3, out_channels=3,
            layers_per_block=2,
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
            block_out_channels=(n_ch, 2*n_ch, 2 * n_ch, 4 * n_ch, 4 * n_ch),
            downsample_type="resnet",
            upsample_type="resnet"
        )

        #noise schedule using DDIM (faster than DDPM)
        self.scheduler = diffusers.schedulers.DDIMScheduler(
            beta_schedule="squaredcos_cap_v2",
            clip_sample=False
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
        '''del residual, noisy_images, noise
        torch.cuda.empty_cache()'''
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

# Main script
if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')  # Set precision for time gain
    model=DiffusionModel_3var()
    '''model_raw = DiffusionModel_3var()
    model = torch.compile(model_raw)''' #compiling the model prior to training can make it muche faster, but can be unstable in some cases, to be developped later

    dataset_train = Dataset_3var_train(l_files, n_files, data_path, 'so_', 'thetao_', 'zos_')
    dataset_val = Dataset_3var_valid(data_path, 'so_val.npy', 'thetao_val.npy', 'zos_val.npy')

    train_loader = DataLoader(dataset_train, batch_size=bsize, shuffle=True, num_workers=0, pin_memory=True) #num_workers should be good but to be verified 
    valid_loader = DataLoader(dataset_val, batch_size=bsize, shuffle=False, num_workers=0, pin_memory=True)

    checkpoint_dir = "f:/StageTH/Code/Diffusion_model/Save" #path to the folder to save your model's checkpoints
    save_loss_callback = SaveLossCallback(save_dir=checkpoint_dir)

    trainer = L.Trainer(
        accumulate_grad_batches=8, #simulates a larger batch to avoid memory constraints
        accelerator="gpu",
        devices=1,
        strategy="auto",  # ou "auto"
        precision="16-mixed",
        max_epochs=n_epochs,
        callbacks=[save_loss_callback],
        default_root_dir=checkpoint_dir,
    )

    trainer.fit(model, train_loader, valid_loader)