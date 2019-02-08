import torch
import torchvision
import numpy as np
from matplotlib import cm
from PIL import Image
from skimage import transform

def get_model(modelname):

    if modelname == 'vgg19':
        model = torchvision.models.vgg19(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        

    elif modelname == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 224,224

        

    elif modelname == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
        model.eval()
        model.cuda()
        model_imsize = 227,227

        

    vgg_mean = (0.485, 0.456, 0.406)
    vgg_std = (0.229, 0.224, 0.225)
    model_mean,model_std = vgg_mean,vgg_std 
    preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(model_imsize),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean = model_mean,std=model_std)
                                           ])
    
    return model, model_imsize, preprocess


def prepare_model_for_gradcam(modelname,model):
    if modelname == 'vgg19':
        last_spatial_layer = model.features._modules['36']
        pass
    elif modelname == 'resnet18':
        last_spatial_layer = model._modules['layer4']._modules['1']._modules['bn2']
        pass
    elif modelname == 'alexnet':
        last_spatial_layer = model._modules['features']._modules['12']
        pass
    def fwdhook(self,input,output):
        self.our_feats = output
    hooked_last_spatial = last_spatial_layer.register_forward_hook(fwdhook)

    def bwdhook(self,grad_in,grad_out):
        self.our_grad_out = grad_out[0]
    bwd_hooked_last_spatial = last_spatial_layer.register_backward_hook(bwdhook)
    
    return last_spatial_layer,hooked_last_spatial,bwd_hooked_last_spatial


def get_saliency(model,ref,last_spatial_layer,class_of_interest,model_imsize,modelname='vgg'):
    ref_scores = model(ref)
    gradcam_loss = torch.sum(ref_scores[:,class_of_interest])
    gradcam_loss.backward()

    Z = 1.
    if modelname == 'vgg19':
        alpha_c_k = (1/Z) * torch.sum(last_spatial_layer.our_grad_out,(2,3))
    #     print(alpha_c_k.shape,tensor_to_numpy(alpha_c_k[0,:10]))
        alpha_into_A = alpha_c_k.unsqueeze(-1).unsqueeze(-1) * last_spatial_layer.our_feats
    elif modelname == 'alexnet':
        pass
    elif modelname == 'resnet18':
        pass
    
    alpha_into_A_channelsum = torch.sum(alpha_into_A,1)
    L_c = torch.nn.functional.relu(alpha_into_A_channelsum)
#     print(L_c.shape)

    L_c_np = tensor_to_numpy(L_c)
    
   
    L_c_np = L_c_np/L_c_np.max((1,2))[:,None,None]
    
    #L_c_np = np.transpose(L_c_np, (1,2,0))
    #heat_map = transform.resize((L_c_np*255).astype(np.uint8),(224,224))
    heat_map = list(map(lambda t:transform.resize((t*255.).astype(np.uint8),model_imsize),L_c_np))
    heat_map = np.array(heat_map)
    #TODO why did i transpose
    #print(heat_map.shape)

    
    return L_c_np,heat_map,ref_scores


def batch_overlay(heat_map,im, model_imsize):
    #TODO Write code for converting a batch of heat mapps into cm.jet images, and overlay them onto the reference image
    heat_map_jet = list(map(lambda h:cm.jet(h),heat_map))
    heat_map_jet = np.array(heat_map_jet)[:,:,:,:3]
    # heat_map_jet = np.transpose(heat_map_jet, (2,0,1,3))
    heat_map_jet_pil = list(map(lambda t: Image.fromarray(np.uint8(t*255)), heat_map_jet))

    #heat_map_jet_pil = Image.fromarray(np.uint8(heat_map))
    #heat_map_jet.shape
    
    saliency_overlayed = []
    for im_i,h in zip(im,heat_map_jet_pil):
        ref_i_pil = Image.fromarray((transform.resize(im_i,model_imsize)*255).astype(np.uint8))
        s = Image.blend(ref_i_pil,h,alpha=0.8)
#         s = np.array(s)
        saliency_overlayed.append(s)
#     saliency_overlayed = np.array(saliency)
    return saliency_overlayed,heat_map_jet_pil

