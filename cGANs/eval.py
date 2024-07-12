import torch
from torch import optim
from torch.utils.data import DataLoader
import config
import utils
from models import Discriminator, Generator, UNet
from dataset import SpaceNetDataset
from tqdm import tqdm
from metrics import compute_losses
import numpy as np

def evaluate(disc, gen, seg, loader):
    disc.eval()
    gen.eval()
    seg.eval()
    with torch.no_grad():
        loop = tqdm(loader, leave=True)
        total_batches = len(loader)
        losses = np.array(
            [0., 0., 0., 0., 0.]
        )
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            input_img = x[:, :3]

            # Generated output
            y_gen = gen(x)

            # Ground truth mask
            gt_mask = x[:, -1].unsqueeze(dim=1)
            
            # Model predicted mask
            pred_mask = seg(y_gen)

            losses += compute_losses(pred_mask, gt_mask)
    losses /= total_batches
    losses = {
        "l1":losses[0], "mse":losses[1], "bce":losses[2], "psnr":losses[3], "f1": losses[4]
    }
    return losses










def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=4).to(config.DEVICE)
    seg = UNet().to(config.DEVICE)

    utils.load_model(config.CHECKPOINT_GEN, gen)
    utils.load_model(config.CHECKPOINT_DISC, disc)
    utils.load_model(config.CHECKPOINT_SEG, seg)

    val_dataset = SpaceNetDataset()
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False)

    """
    Loss metrics:
    1. MAE/L1
    2. MSE
    3. BCE (Segmentation Map)
    4. IoU
    5. PSNR
    6. F1 Score
    """

    losses = evaluate(disc, gen, seg, val_loader)
    np.save(config.METRICS_SAVE_PATH, losses)
    print("Metrics and associated losses: ")
    print(losses)


if __name__ == "__main__":
    main()

