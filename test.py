import argparse
import os
import numpy as np
import math
import glob
import torchvision.transforms as transforms
# from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from datasets import *
from models import *
from models.balle2018.model import BMSHJ2018Model
from models.balle2017.model import BLS2017Model
from models.distributed_model import HyperPriorDistributedAutoEncoder, DistributedAutoEncoder

import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml
#from models2 import *
from swinIR import SwinIR
#from NAFNet import *
# import BFBatchNorm2d
# import torch.nn.init as init
# from RidNet import RIDNET
# from mwcnn import MWCNN
# from CGNet import CascadedGaze

import PIL.Image as Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from datasets import Vimeo90K_interp
from pytorch_msssim import ms_ssim
import math

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="configuration")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
# parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=32, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")

opt = parser.parse_args()
print(opt)
# opt.add_argument('--config', type=str, default='configs/config.yaml', help="configuration")

cuda = True if torch.cuda.is_available() else False

input_shape = (3, 64, 64)
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
hr_shape = (144, 176)
# Loss function
adversarial_loss = torch.nn.MSELoss()

class ImageDataset(Dataset):
    def __init__(self, root, root2, hr_shape):
        hr_height, hr_width = hr_shape
        # Transforms for low resolution images and high resolution images
        self.SI_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 2, hr_width // 2), Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(), 
            ]
        )
        self.decoded_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height // 2, hr_width // 2), Image.BICUBIC),
                #transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )
        self.real_transform = transforms.Compose(
            [
                #transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"),key=numericalSort)
        self.files2 = sorted(glob.glob(root2 + "/*.*"),key=numericalSort)
        #self.files3 = sorted(glob.glob(root3 + "/*.*"),key=numericalSort)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img2 = Image.open(self.files2[index % len(self.files)])
        #img3 = Image.open(self.files3[index % len(self.files)])
        #img2 = img2.convert("L")
        #img = img.convert("L")
        img_real = self.real_transform(img)
        img_SI = self.SI_transform(img2)
        #img_real = self.real_transform(img3)

        return {"real": img_real,"SI": img_SI}
    
    def __len__(self):
        return len(self.files)


img_channel = 1
width = 8

# Initialize generator and discriminator
generator = SwinIR(upscale=1, img_size=(64, 64),
                window_size=8, img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='jpeg')#GeneratorResNet(input_shape, 3)



model_class = DistributedAutoEncoder
model = model_class(num_filters=192)
model = model.cuda() 

                
def map_layers(weight):
    """ Since the pre-trained weights provided for bls17 by us were trained with
        different layer names, we map the layer names in the state dictionaries
        to the new names using the following function map_layers().
    """
    return OrderedDict([(k.replace('z', 'w'), v) if 'z' in k else (k, v) for k, v in weight.items()])

# generator = MWCNN()

generator.cuda()
# generator.load_state_dict(torch.load("outputs/weight/swinproposedGRAY0013.pt"))


checkpoint = torch.load('./outputs/weight/swinproposedGRAY0007.pt', map_location=torch.device('cuda' if cuda else 'cpu'))

checkpoint['model_state_dict'] = map_layers(checkpoint['model_state_dict'])
generator.load_state_dict(checkpoint['model_state_dict'])


checkpoint = torch.load('./outputs/weight/proposed_013_025_grayscal_wo_CAM.pt', map_location=torch.device('cuda' if cuda else 'cpu'))

checkpoint['model_state_dict'] = map_layers(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizer.param_groups[0]['capturable'] = True

if cuda:
    generator.cuda()
    # discriminator.cuda()
    adversarial_loss.cuda()

def get_bpp(model_out, config):  # Returns calculated bpp for train and test
    alpha = config['alpha']
    beta = config['beta']
    if config['baseline_model'] == 'bmshj18':
        if config['use_side_info']:  # If the side information (correlated image) has to be used
            ''' 
            The loss function consists of:
            Rate terms for input image (likelihoods), correlated image (y_likelihoods),
            and the common information (w_likelihoods), hyperpriors for input image (z_likelihoods)
            , hyperpriors for correlated image (z_likelihoods_cor).
            Sum of these rate terms is returned as bpp, along with the actual bpp transmitted over the channel,
            which consists only of likelihoods + z_likelihoods.
            '''
            x_recon, likelihoods, z_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            # bpp += alpha * (torch.sum(torch.log(y_likelihoods)) + torch.sum(torch.log(z_likelihoods_cor))) / size_est
            # bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
            return bpp, transmitted_bpp
        else:  # The baseline implementation (Balle2018) without the side information
            x_recon, likelihoods, z_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            return bpp, bpp
    elif config['baseline_model'] == 'bls17':
        if config['use_side_info']:
            x_recon, likelihoods= model_out
            size_est = (-np.log(2) * x_recon.numel() / 1)
            bpp = torch.sum(torch.log(likelihoods)) / size_est

            return bpp, bpp
        else:
            x_recon, likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            return bpp, bpp
    return None


def get_distortion(config, out, img, cor_img, mse):
    distortion = None
    alpha = config['alpha']
    if config['use_side_info']:
        ''' 
        The loss function consists of:
        Distortion terms for input image (x_recon), and correlated image (x_cor_recon).
        '''
        #x_recon, y_recon = out[0], out[1]
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), out.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
            distortion += alpha * (1 - ms_ssim(cor_img.cpu(), out.cpu(), data_range=1.0, size_average=True,
                                               win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, out)
            # distortion += alpha * mse(cor_img, y_recon)
    else:
        x_recon = out[0]
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, x_recon)

    return distortion


def map_layers(weight):
    """ Since the pre-trained weights provided for bls17 by us were trained with
        different layer names, we map the layer names in the state dictionaries
        to the new names using the following function map_layers().
    """
    return OrderedDict([(k.replace('z', 'w'), v) if 'z' in k else (k, v) for k, v in weight.items()])


def save_image(x_recon, img, SI, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().detach().numpy(), 0, 255)
    img = np.clip((img * 255).squeeze().cpu().detach().numpy(), 0, 255)
    # print(img_recon.shape)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    # img_recon = img_recon.astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    # img = img.astype('uint8')
    img_SI = np.clip((SI * 255).squeeze().cpu().detach().numpy(), 0, 255)
    img_SI = np.transpose(img_SI, (1, 2, 0)).astype('uint8')
    # img_SI = img_SI.astype('uint8')
    # img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
    img_final = Image.fromarray(np.concatenate((img, img_recon, img_SI), axis=1), 'RGB')
    # img_final = Image.fromarray(np.concatenate((img, img_recon, img_SI), axis=1))
    if not os.path.exists(path):
        os.makedirs(path)
    img_final.save(os.path.join(path, name + '.png'))

def save_image2(x_recon, x, SI, path, name):
    """
    Save grayscale images: reconstructed, original, and side information.

    Parameters:
        x_recon (np.ndarray): Reconstructed grayscale image.
        x (np.ndarray): Original grayscale image.
        SI (np.ndarray): Side information image.
        path (str): Directory to save images.
        name (str): Base name for saved images.
    """
    # Ensure output directory exists
    os.makedirs(path, exist_ok=True)

    # Function to normalize and convert image to uint8
    def to_uint8(img):
        if img.dtype != np.uint8:
            img = np.clip((img * 255).squeeze().cpu().numpy(), 0, 255)
            img = img.astype(np.uint8)
        return img

    # Convert and save each image
    #Image.fromarray(to_uint8(x)).save(os.path.join(path, f"{name}_original.png"))
    Image.fromarray(to_uint8(x_recon)).save(os.path.join(path, f"{name}.png"))
    #Image.fromarray(to_uint8(SI)).save(os.path.join(path, f"{name}_sideinfo.png"))

def split_tensor(tensor, tile_size=256):
    mask = torch.ones_like(tensor)
    # use torch.nn.Unfold
    stride  = tile_size
    unfold  = torch.nn.Unfold(kernel_size=(tile_size, tile_size), stride=stride)
    # Apply to mask and original image
    mask_p  = unfold(mask)
    patches = unfold(tensor)
	
    patches = patches.reshape(3, tile_size, tile_size, -1).permute(3, 0, 1, 2)
    if tensor.is_cuda:
        patches_base = torch.zeros(patches.size(), device=tensor.get_device())
    else: 
        patches_base = torch.zeros(patches.size())
	
    tiles = []
    for t in range(patches.size(0)):
         tiles.append(patches[[t], :, :, :])
    return tiles, mask_p, patches_base, (tensor.size(2), tensor.size(3))



def rebuild_tensor(tensor_list, mask_t, base_tensor, t_size, tile_size=256):
    stride  = tile_size  
    # base_tensor here is used as a container

    for t, tile in enumerate(tensor_list):
         # print(tile.size())
         base_tensor[[t], :, :] = tile  
	 
    base_tensor = base_tensor.permute(1, 2, 3, 0).reshape(3*tile_size*tile_size, base_tensor.size(0)).unsqueeze(0)
    # print(base_tensor.shape)
    fold = torch.nn.Fold(output_size=(t_size[0], t_size[1]), kernel_size=(tile_size, tile_size), stride=stride)
    # print(fold(base_tensor))
    # https://discuss.pytorch.org/t/seemlessly-blending-tensors-together/65235/2?u=bowenroom
    output_tensor = fold(base_tensor)/fold(mask_t)
    #output_tensor = fold(base_tensor)
    return output_tensor


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def apply_random_mask(imgs):
    idx = np.random.randint(0, opt.img_size - opt.mask_size, (imgs.shape[0], 2))

    masked_imgs = imgs.clone()
    for i, (y1, x1) in enumerate(idx):
        y2, x2 = y1 + opt.mask_size, x1 + opt.mask_size
        masked_imgs[i, :, y1:y2, x1:x2] = -1

    return masked_imgs


val_loader =DataLoader(ImageDataset("hallWZsource","hallSIQ32", hr_shape=hr_shape), batch_size=1)


print('read datasets')

print('read datasets finish...!!!')
saved_samples = {}




mse = torch.nn.MSELoss(reduction='mean')
mse = mse.cuda() if True else mse
with open(opt.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
# experiment_name = model_class.__name__ + '_' + str(train_dataset) + '_' + config['distortion_loss'] + '_lambda:' + \
#                       str(config['lambda'])
experiment_name = 'proposed'

print('Experiment: ', experiment_name)

weight_folder = None
if config['save_weights']:
        weight_folder = os.path.join('outputs/', 'weight')
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

min_val_loss = None



    # Validation
model.eval()
val_loss = []
val_mse = []
val_mseSI = []
val_mse2 = []
    # val_msssim = []
val_bpp = []
val_transmitted_bpp = []
val_distortion = []
psnrlist =[]
with torch.no_grad():
        for i, data in enumerate(iter(val_loader)):
            
            imgsi = Variable(data["SI"].type(Tensor))
            #imgs_decoded = Variable(data["decoded"].type(Tensor))
            img = Variable(data["real"].type(Tensor))

            
                #if (i%3)==0:
            out = model(img, imgsi)
                #else:
                 #   out = model(img, cor_img)
       
            bpp, transmitted_bpp = get_bpp(out, config)

            x_recon = out[0]
            
            gen_imgs = generator(x_recon, imgsi)
            
            mse_dist = mse(img, gen_imgs)
            mse_distSI = mse(img, imgsi)
            mse_dist1 = mse(img, x_recon)
            # msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                  # win_size=7)
            # msssim_db = -10 * np.log10(msssim)
            
            likelihoods = out[1]
            size_est = (-np.log(2) * x_recon.numel() / 1)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            

            distortion = get_distortion(config, gen_imgs, img, img, mse)

            loss = distortion  # multiplied by (255 ** 2) for distortion scaling

            val_mse.append(mse_dist.item())
            val_mseSI.append(mse_distSI.item())
            val_mse2.append(mse_dist1.item())
            val_bpp.append(bpp.item())
            val_transmitted_bpp.append(transmitted_bpp.item())
            val_loss.append(loss.item())
            # val_msssim.append(msssim_db.item())
            val_distortion.append(distortion.item())
            print(10 * np.log10(1 / (sum(val_mse) / (len(val_mse)))))
            print(bpp)
            psnrlist.append( 10 * np.log10(1 /val_mse[i]))
            # print('bpp :', bpp)
            if i<100:
                if config['save_image']:
                    results_path = os.path.join(config['save_output_path'], 'results')
                    save_image2(gen_imgs,img,imgsi, os.path.join(results_path, '{}_images'.format(experiment_name)),
                                str(i))

val_loss_to_track = 10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))


    # Verbose
tracking = [
                    'Loss = {:.4f},'.format(val_loss_to_track),
                    'BPP = {:.8f},'.format(sum(val_bpp) / len(val_bpp)),
                    'Distortion = {:.4f},'.format(sum(val_distortion) / len(val_distortion)),
                    'Transmitted BPP = {:.8f},'.format(sum(val_transmitted_bpp) / len(val_transmitted_bpp)),
                    'PSNR swin = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))),
                    'PSNR AE = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse2) / (len(val_mse2))))),
                    'PSNR SI = {:.4f}'.format(10 * np.log10(1/(sum(val_mseSI)/(len(val_mseSI)))))]
print(" ".join(tracking))

psnrEnh = 10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))
psnrSI = 10 * np.log10(1/(sum(val_mseSI)/(len(val_mseSI))))
psnrAE = 10 * np.log10(1 / (sum(val_mse2) / (len(val_mse2)))) 
bpptot = sum(val_bpp) / len(val_bpp)

output_file = "lambda007Q32.dat"

with open(output_file, 'w') as f:
    f.write("Frame\tPSNR\tBPP \n")
    for i, (psnr, rate) in enumerate(zip(psnrlist, val_bpp), start=1):
        f.write(f"{i}\t{psnr:.2f}\t{rate}\n")
    f.write("\n PsnrSI \t\t PsnrEnhance \t\t PsnrAE \t\t BPPToT \n")
    f.write(f"{psnrSI}\t{psnrEnh}\t{psnrAE}\t{bpptot}")


print(f"Results saved to {output_file}")

    # Save weights
# if config['save_weights']:
#         if min_val_loss is None or min_val_loss < val_loss_to_track:
#             min_val_loss = val_loss_to_track
#             save_path = os.path.join(weight_folder, 'swinproposedGRAY0013.pt')
#             torch.save({
#                 "model_state_dict": generator.state_dict(),
#                 "optimizer_state_dict": optimizer_G.state_dict(),
#             }, save_path)
#             print('update_weights')

