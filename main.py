import argparse
from PIL import Image
import torch
import torchvision
import numpy as np
import wget
from matplotlib import cm
import matplotlib.pyplot as plt
from skimage import io, transform
import skimage

import gradcam

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="vgg19", choices=["vgg19", "resnet18", "alexnet"], help="Choose architecture(default=vgg19)")
    parser.add_argument("--coi", default=245, type=int, help="The class on which you want to see the output(default=245)")
    parser.add_argument("--gpu", action='store_true', help="Flag to use a GPU")
    return parser.parse_args()


def get_image(preprocess):
    #downloading image
    wget.download('https://raw.githubusercontent.com/ramprs/grad-cam/master/images/cat_dog.jpg')
    #reading the image
    image = io.imread('cat_dog.jpg')
    #the pixels has range from (0 - 1). making them to the range of (0-255) and transforming them to the Image type
    image_pil = Image.fromarray(np.uint8(image*255))

    #get the transformed image
    transformed_image = preprocess(image_pil).requires_grad_(True)
    #adding the first dimension as batch and converting to cuda
    transformed_image = transformed_image.unsqueeze(0).cuda()

    return transformed_image, image
##########
in_arg = get_input_args()
model, model_imsize, preprocess = gradcam.get_model(in_arg.arch)
last_spatial_layer,hooked_last_spatial,bwd_hooked_last_spatial = gradcam.prepare_model_for_gradcam(in_arg.arch, model)

ref, image = get_image(preprocess)

L_c_np,heat_map,ref_scores = gradcam.get_saliency(model, torch.cat([ref],0), last_spatial_layer, in_arg.coi, model_imsize, in_arg.arch)
saliency_overlayed,heat_map_jet_pil = gradcam.batch_overlay(heat_map,[image], model_imsize)

plt.imshow(saliency_overlayed[0])

