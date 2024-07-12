
This is a guide for training and utilizing the ControlNet pipeline provided in the HuggingFace Diffusers pipeline.
This guide serves to give an explainer on how to adapt this guide to be able to train and run the ControlNet model on the SpaceNet dataset. Anyone wishing to implement the ControlNet model should first go through it and try to adapt it according to the nature of their data.
Follow the guide as it is to set up the diffusers library and prepare your environment with the necessary requirements. 

###Replace Files###
In the accompanying repository to this guide, you will find two files by the name of Spacenet.py and train_controlnet.py. Download both of these and navigate to diffusers/examples/controlnet. Now paste both files here, replacing any files of the same name. 
The Spacenet.py file creates a HuggingFace Dataset template which has three things: images, conditions, and prompts. Since there are no prompts available for the Spacenet Dataset, they are treated as empty strings. This does not take away from the Stable Diffusion model’s own textual reasoning capabilities and you will be able to provide prompts when generating your own images.

Dataset
To prepare for training, create a data folder in the controlnet directory. This folder should have two subfolders, conditions and images. You should now add all your input images to the images folder and all the conditioning images (segmentation masks) to the conditions folder. It is vital that every unique image in the images folder and its corresponding mask in the conditions folder should have identical filenames. You are now ready to train your model.

Training
To train your model, follow the same instructions as given in the guide with minor changes. If you experience “CUDA out of memory” problems, use the instructions given for 12GB GPUs provided on the link. This may require your to install the xformers library if you haven’t already. The biggest change that you will have to make is to replace
--dataset_name=fusing/fill50k 

with

--dataset_name=Spacenet.py 

Make sure that you have replaced the train_controlnet.py file.

Using the Model
The test.ipynb file demonstrates how you may use the trained model to generate your own images. Be sure to change the file directories according to your system configuration and provide prompts as required. There is one thing to note here. There are two versions of Stable Diffusion that can be used as a backbone:

 

base_model_path refers to a vanilla version of Stable Diffusion 1.5 whereas remote_sensing_model refers to a version of it that has been finetuned for remote sensing applications. You may choose either one of these for your experiments. To choose the vanilla version, simply replace remote_sensing_model with base_model_path in the next code cell:

 

Running this model requires around 5-10 GB to download all necessary weights for the pipeline.
