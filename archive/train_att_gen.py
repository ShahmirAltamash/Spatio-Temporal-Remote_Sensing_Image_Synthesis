import torch
import config
import utils
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import sys
from dataset import SpaceNetDataset
from models import Discriminator, Att_Generator
from torchvision.utils import save_image
import matplotlib.pyplot as plt

def train_fn(disc, gen, loader, opt_disc, opt_gen, l1, bce, g_scaler, d_scaler, writer, epoch):
    loop = tqdm(loader, leave=True)


    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        input_img = x[:, 0:3]
        # Train Discriminator
        with torch.cuda.amp.autocast():
            #print(x.shape)
            y_fake = gen(x)
            D_real = disc(input_img, y)
            D_fake = disc(input_img, y_fake.detach())
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(input_img, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

    print(f"Epoch = {epoch + 1}")
    print(f"G Loss = {G_loss:.4f}")
    print(f"D Loss = {D_loss:.4f}")

    return D_loss.item(), G_loss.item()


def validate_fn(disc, gen, loader, mse, writer, epoch):
    loop = tqdm(loader, leave=True)
    MSE_gen = 0.
    with torch.no_grad():
        for idx, (x, y) in enumerate(loop):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            input_img = x[:, 0:3]
            y_fake = gen(x)
            MSE_gen += mse(y_fake, y)
    MSE_gen = MSE_gen/len(loop)
    print(f"Epoch = {epoch + 1}")
    print(f"Generator MSE Loss (Validation): {MSE_gen:.4f}")
    return MSE_gen.item()

        



def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Att_Generator(in_channels=4).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()
    MSE_LOSS = nn.MSELoss()
    MIN_LOSS_G = 10e6

    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_ATT_GEN, gen, opt_gen, config.LEARNING_RATE)
        utils.load_checkpoint(config.CHECKPOINT_ATT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = SpaceNetDataset(train=0)
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    val_dataset = SpaceNetDataset(train=1)
    val_loader = DataLoader(val_dataset, config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

   
    fixed_inputs, fixed_targets = next(iter(val_loader))
    fixed_inputs, fixed_targets = fixed_inputs[:1].to(config.DEVICE), fixed_targets[:1].to(config.DEVICE)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter()
    # writer.add_graph(disc)
    # writer.add_graph(gen)

    d_loss_history = []
    g_loss_history = []

    g_loss_val_history = []
    threshold = 0.029
    epoch_threshold = 20

    step = 0

    for epoch in range(config.NUM_EPOCHS):
        d_loss, g_loss = train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler, writer, epoch)
        
        d_loss_history.append(d_loss)
        g_loss_history.append(g_loss)

        if config.VALIDATE:
            g_loss_val = validate_fn(disc, gen, val_loader, MSE_LOSS, writer, epoch)
            g_loss_val_history.append(g_loss_val)
            np.save("Loss History/att_generator_best_val.npy", g_loss_val_history)

            if g_loss_val < MIN_LOSS_G:
                MIN_LOSS_G = g_loss_val
                utils.save_model(gen, config.CHECKPOINT_ATT_GEN_VAL_BEST)
            if (g_loss_val < threshold) & (epoch > epoch_threshold):
                utils.save_model(gen, config.CHECKPOINT_ATT_GEN_VAL_BEST)
                print(f'Validation loss is less than threshold at epoch {epoch}')
                print(f'closing the training loop at epoch {epoch}')
        

# Saving the model state and dictionary after every epoch so that we can retreive parameters at the best epoch
        if config.SAVE_MODEL:
            utils.save_model(gen, filename=f"Saved Models/att_gen_best_epoch_{epoch}.pth.tar")
            utils.save_model(disc, filename=f"Saved Models/att_disc_best_epoch_{epoch}.pth.tar")
            utils.save_some_examples(gen, val_loader, epoch, folder="Evaluation")
        np.save("Loss History/att_discriminator_best_loss.npy", d_loss_history)
        np.save("Loss History/att_generator_best_loss.npy", g_loss_history)
           
        with torch.no_grad():
            fixed_gen = gen(fixed_inputs)

        fixed_target = make_grid(fixed_targets)
        fixed_generation = make_grid(fixed_gen)

        
       

        writer.add_image("Targets", fixed_target, global_step=epoch)
        writer.add_image("Generated", fixed_generation, global_step=epoch)
        if (epoch + 1) % 1 == 0:  # Every 10 epochs

            #print(f'fixed targets shape: {fixed_targets.shape}')
            #print(f'fixed generation shape: {fixed_gen.shape}')
            fixed_targets_np = fixed_targets[0:1,:,:,:].cpu().detach().permute(0,2,3,1).numpy()
            fixed_generation_np = fixed_gen[0:1,:,:,:].cpu().detach().permute(0,2,3,1).numpy()

             # Save images
            save_path = 'Att_Gen_train_images_best/'  
           
            save_image(fixed_targets, f'{save_path}epoch_{epoch+1}_targets.png', normalize=True)
            save_image(fixed_generation, f'{save_path}epoch_{epoch+1}_generated.png', normalize=True)
          
   
        writer.add_scalars("Loss/", {
        "Discriminator": d_loss,
        "Generator": g_loss
        }, global_step=epoch)

        if config.VALIDATE:
            writer.add_scalars("Validation Loss/", {
                "Generator": g_loss_val 
            }, global_step=epoch)


    np.save("Loss History/att_discriminator_best_loss.npy", d_loss_history)
    np.save("Loss History/att_generator_best_loss.npy", g_loss_history)


    if config.VALIDATE:
        np.save("Loss History/att_generator_best_val.npy", g_loss_val_history)
       
if __name__ == "__main__":
    main()