# ControlNet for SpaceNet Dataset

This guide explains how to train and utilize the [ControlNet](https://huggingface.co/docs/diffusers/en/using-diffusers/controlnet) pipeline provided in the HuggingFace Diffusers library, specifically adapted for the SpaceNet dataset.

## Getting Started

First, follow the [official ControlNet guide](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README.md) to set up the diffusers library and prepare your environment with the necessary requirements.

## File Replacement

In this repository, you'll find two files:
- `Spacenet.py`
- `train_controlnet.py`

Download both files and navigate to `diffusers/examples/controlnet`. Replace any existing files with the same names.

The `Spacenet.py` file creates a HuggingFace Dataset template with three components: images, conditions, and prompts. Since the SpaceNet Dataset doesn't include prompts, they are treated as empty strings. This doesn't affect the Stable Diffusion model's textual reasoning capabilities, and you can provide prompts when generating your own images.

## Dataset Preparation

1. Create a `data` folder in the `controlnet` directory.
2. Inside `data`, create two subfolders: `conditions` and `images`.
3. Add all input images to the `images` folder.
4. Add all conditioning images (segmentation masks) to the `conditions` folder.

**Important**: Ensure that each unique image in the `images` folder and its corresponding mask in the `conditions` folder have identical filenames.

## Training

Follow the instructions in the official guide with these modifications:

1. If you encounter "CUDA out of memory" issues, use the instructions for 12GB GPUs provided in the link. This may require installing the `xformers` library.
2. Replace `--dataset_name=fusing/fill50k` with `--dataset_name=Spacenet.py`.

Make sure you've replaced the `train_controlnet.py` file as mentioned earlier.

## Using the Model

The `test.ipynb` file demonstrates how to use the trained model to generate your own images. Adjust file directories according to your system configuration and provide prompts as needed.

### Stable Diffusion Backbone

There are two versions of Stable Diffusion that can be used as a backbone:

![Stable Diffusion Versions](images/stable_diffusion_versions.jpeg)

`base_model_path` refers to a vanilla version of Stable Diffusion 1.5 whereas `remote_sensing_model` refers to a version of it that has been finetuned for remote sensing applications. You may choose either one of these for your experiments. To choose the vanilla version, simply replace `remote_sensing_model` with `base_model_path` in the next code cell:

![Code Cell Example](images/code_cell_example.jpeg)

**Note**: Running this model requires approximately 5-10 GB to download all necessary weights for the pipeline.

## Additional Information

For more detailed information or if you encounter any issues, please refer to the original ControlNet documentation or open an issue in this repository.
