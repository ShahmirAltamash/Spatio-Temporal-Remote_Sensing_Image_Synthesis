import torch
from pathlib import Path
from torchvision import transforms

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 0
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
VALIDATE = True

CHECKPOINT_DISC = "Saved Models/disc2.pth.tar"
CHECKPOINT_ATT_DISC = "Saved Models/att_disc.pth.tar"
CHECKPOINT_ATT_DISC_TRIAL = "Saved Models/att_disc_trial.pth.tar"
CHECKPOINT_ATT_DISC_BEST = "Saved Models/att_disc_best.pth.tar"

CHECKPOINT_GEN = "Saved Models/gen2.pth.tar"
CHECKPOINT_ATT_GEN = "Saved Models/att_gen.pth.tar"
CHECKPOINT_ATT_GEN_TRIAL = "Saved Models/att_gen_trial.pth.tar"
CHECKPOINT_ATT_GEN_BEST = "Saved Models/att_gen_best.pth.tar"

CHECKPOINT_SEG = "Saved Models/seg_model.pth"

CHECKPOINT_GEN_VAL = "Saved Models/gen_best_val2.pth.tar"
CHECKPOINT_ATT_GEN_VAL = "Saved Models/att_gen_best_val.pth.tar"
CHECKPOINT_ATT_GEN_VAL_TRIAL = "Saved Models/att_gen_best_val_trial.pth.tar"
CHECKPOINT_ATT_GEN_VAL_BEST = "Saved Models/att_gen_best_val_best.pth.tar"

TRAIN_CSV_DIR = Path("output_csvs/train_dataset.csv")
TRAIN_DIR = Path("SN7_buildings_train/train")
EVAL_DIR = Path('SN7_buildings_test_public/test_public')
TEST_CSV_DIR = Path('output_csvs/test_dataset.csv')
EVAL_CSV_DIR = Path('output_csvs/df_test_untidy.csv')
GEN_TEST_DIR =Path('SN7_generated_test_images')
VAL_CSV_DIR = Path('output_csvs/val_dataset.csv')
MASK_DIR_TRAIN = Path("SN7_seg_masks/SN7_seg_masks")
METRICS_SAVE_PATH = Path("Loss History/Metrics/out_metrics.npy")



IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=None),
    # transforms.Normalize(
    #     mean=(0.5,), std=(0.5,)
    # )
])

MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512), antialias=None)
])