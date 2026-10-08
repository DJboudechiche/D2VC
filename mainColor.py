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
#from CGANmodel import *
#from models2 import *
from swinIR import SwinIR
from NAFNet import *
# import BFBatchNorm2d
# import torch.nn.init as init
#from colorUnet import KernelEstimation
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
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
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
Enhance = SwinIR(upscale=1, img_size=(64, 64),
                window_size=8, img_range=1., depths=[6, 6, 6, 6],
                embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='jpeg')#GeneratorResNet(input_shape, 3)

Enhance = Enhance.cuda()
# generator = Generator(input_shape)
# discriminator = Discriminator(input_shape)




model_class = DistributedAutoEncoder
model = model_class(num_filters=192)
model = model.cuda() 

# class DnCNN(nn.Module):
# 	"""DnCNN as defined in https://arxiv.org/abs/1608.03981 
# 	   reference implementation: https://github.com/SaoYan/DnCNN-PyTorch"""
# 	def __init__(self, depth=20, n_channels=64, image_channels=3, bias=False, kernel_size=3):
# 		super(DnCNN, self).__init__()
# 		kernel_size = 3
# 		padding = 1

# 		self.bias = bias;
# 		if not bias:
# 			norm_layer = BFBatchNorm2d.BFBatchNorm2d
# 		else:
# 			norm_layer = nn.BatchNorm2d
# 		self.depth = depth;


# 		self.first_layer = nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)

# 		self.hidden_layer_list = [None] * (self.depth - 2);
# 		
# 		self.bn_layer_list = [None] * (self.depth -2 );
# 		
# 		for i in range(self.depth-2):
# 			self.hidden_layer_list[i] = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=self.bias);
# 			self.bn_layer_list[i] = norm_layer(n_channels)
# 		
# 		self.hidden_layer_list = nn.ModuleList(self.hidden_layer_list);
# 		self.bn_layer_list = nn.ModuleList(self.bn_layer_list);
# 		self.last_layer = nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=self.bias)
# 		
# 		self._initialize_weights()

# 	@staticmethod
# 	def add_args(parser):
# 		"""Add model-specific arguments to the parser."""
# 		parser.add_argument("--in-channels", type=int, default=1, help="number of channels")
# 		parser.add_argument("--hidden-size", type=int, default=64, help="hidden dimension")
# 		parser.add_argument("--num-layers", default=20, type=int, help="number of layers")
# 		parser.add_argument("--bias", action='store_true', help="use residual bias")

# 	@classmethod
# 	def build_model(cls, args):
# 		return cls(image_channels = args.in_channels, n_channels = args.hidden_size, depth = args.num_layers, bias=args.bias)

# 	def forward(self, x):
# 		y = x
# 		out = self.first_layer(x);
# 		out = F.relu(out);

# 		for i in range(self.depth-2):
# 			out = self.hidden_layer_list[i](out);
# 			out = self.bn_layer_list[i](out);
# 			out = F.relu(out)

# 		out = self.last_layer(out);
# 		
# 		return y-out

# 	def _initialize_weights(self):
# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d):
# 				init.kaiming_normal_(m.weight, a=0, mode='fan_in')
# 				if m.bias is not None:
# 					init.constant_(m.bias, 0)
# 			elif isinstance(m, nn.BatchNorm2d) or isinstance(m, BFBatchNorm2d.BFBatchNorm2d):
# 				m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
# 				init.constant_(m.bias, 0)

# class ARCNN(nn.Module):
#     def __init__(self):
#         super(ARCNN, self).__init__()
#         self.base = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU(),
#             nn.Conv2d(64, 32, kernel_size=7, padding=3),
#             nn.PReLU(),
#             nn.Conv2d(32, 16, kernel_size=1),
#             nn.PReLU()
#         )
#         self.last = nn.Conv2d(16, 3, kernel_size=5, padding=2)

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight, std=0.001)

#     def forward(self, x):
#         x = self.base(x)
#         x = self.last(x)
#         return x
                
def map_layers(weight):
    """ Since the pre-trained weights provided for bls17 by us were trained with
        different layer names, we map the layer names in the state dictionaries
        to the new names using the following function map_layers().
    """
    return OrderedDict([(k.replace('z', 'w'), v) if 'z' in k else (k, v) for k, v in weight.items()])

# generator = MWCNN()

checkpoint = torch.load('./outputs/weight/swinproposedGRAY0007.pt', map_location=torch.device('cuda' if cuda else 'cpu'))

checkpoint['model_state_dict'] = map_layers(checkpoint['model_state_dict'])
Enhance.load_state_dict(checkpoint['model_state_dict'])

checkpoint = torch.load('./outputs/weight/proposed_0026_0067_grayscal_wo_CAM.pt', map_location=torch.device('cuda' if cuda else 'cpu'))

checkpoint['model_state_dict'] = map_layers(checkpoint['model_state_dict'])
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# optimizer.param_groups[0]['capturable'] = True


generator = KernelEstimation(kernel_size =3)

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
            # transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            # bpp += alpha * torch.sum(torch.log(y_likelihoods)) / size_est
            # bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
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
    Image.fromarray(to_uint8(x)).save(os.path.join(path, f"{name}_original.png"))
    Image.fromarray(to_uint8(x_recon)).save(os.path.join(path, f"{name}_reconstructed.png"))
    Image.fromarray(to_uint8(SI)).save(os.path.join(path, f"{name}_sideinfo.png"))

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


# def save_sample(saved_samples):
    # Generate inpainted image
    # gen_imgs = generator(out, saved_samples["lowres"])
    # Save sample
    # sample = torch.cat((saved_samples["masked"].data, gen_imgs.data, saved_samples["imgs"].data), -2)
    # save_image(sample, "images/%d.png" % batches_done, nrow=5, normalize=True)


path = 'vimeo90k'

train_dataset, val_dataset = Vimeo90K_interp('vimeo_triplet')
#val_loader =DataLoader(ImageDataset("foremanWZsource","foremanSIQ38", hr_shape=hr_shape), batch_size=1)


print('read datasets')
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)#, shuffle=True, num_workers=3)
# test_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=3)
print('read datasets finish...!!!')
saved_samples = {}




mse = torch.nn.MSELoss(reduction='mean')
mse = mse.cuda() if True else mse
with open(opt.config, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
experiment_name = model_class.__name__ + '_' + str(train_dataset) + '_' + config['distortion_loss'] + '_lambda:' + \
                      str(config['lambda'])
experiment_name = 'proposed'

print('Experiment: ', experiment_name)

weight_folder = None
if config['save_weights']:
        weight_folder = os.path.join('outputs/', 'weight')
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

min_val_loss = None

for epoch in range(opt.n_epochs):
    print('epoch:', epoch)
    for i, (imgscolor,imgsSIcolor, imgref, im2color, imSIcolor, im1color) in enumerate(train_loader):
        # print(i)
        # print('')
        # imgsgray = batch[0]
        # imgsSIgray = batch[1]
        # imgscolor = batch[2]
        # imgsSIcolor = batch[3]

        # masked_imgs = apply_random_mask(imgscolor)

        # Adversarial ground truths
        # valid = Variable(Tensor(imgscolor.shape[0], *discriminator.output_shape).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(imgscolor.shape[0], *discriminator.output_shape).fill_(0.0), requires_grad=False)

        # print(valid.shape)
        if cuda:
            # imgsgray = imgsgray.type(Tensor)
            imgref = imgref.type(Tensor)
            imgscolor = imgscolor.type(Tensor)
            imgsSIcolor = imgsSIcolor.type(Tensor)
            im2color = im2color.type(Tensor)
            imSIcolor = imSIcolor.type(Tensor)
            im1color = im1color.type(Tensor)
            # masked_imgs = masked_imgs.type(Tensor)

        # imgsgray = Variable(imgsgray)
        imgref = Variable(imgref)
        imgscolor = Variable(imgscolor)
        imgsSIcolor = Variable(imgsSIcolor)
        im2color = Variable(im2color)
        imSIcolor = Variable(imSIcolor)
        im1color = Variable(im1color)
        # masked_imgs = Variable(masked_imgs)
      
             # frame11 = torch.Tensor(tile_tensors_WZ[t]).cuda()
             # imgSI1 = torch.Tensor(tile_tensors_SI[t]).cuda()
             
        #out, _ = model.forward(imgscolor,imgsSIcolor)

        # out, _ = model.forward(imgscolor)
           
        
        with torch.no_grad():
                out = model(imgscolor, imgsSIcolor)            
        
        out = out[0]
        
        with torch.no_grad():
                out = Enhance(out, imgsSIcolor)  
        
        #out = out[0]
        
        #print(out.shape)
        
        optimizer_G.zero_grad()
        # print(out.shape)
        if (i%3)==0:
           gen_imgs = generator(out, im1color)
        else:
           gen_imgs = generator(out, imSIcolor)

        # gen_imgs = generator(out, imSIcolor)


        # Generate a batch of images
        # gen_imgs = generator(out,imgsSIcolor)
        # gen_imgs = generator(out)
        
        MSE1 = mse(out, imgscolor)
        MSE2 = mse(gen_imgs, im2color)
        MSE3 = mse(gen_imgs, imSIcolor)
        # print(gen_imgs.shape)

        g1_loss =  adversarial_loss(gen_imgs, im2color)
        
        g_loss =  g1_loss

        g_loss.backward()
        optimizer_G.step()
        
        
        
        if i%500==0 :
           print(
               "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [PSNRgray: %f] [PSNRcolor: %f] [PSNRSIcolor: %f]"
               % (epoch, opt.n_epochs, i, len(train_loader), g_loss.item(), 10 * np.log10(1 / MSE1.item()),10 * np.log10(1 / MSE2.item()), 10 * np.log10(1 / MSE3.item()))  
            )
        # if i>12700:
        #     results_path = os.path.join('./outputs', 'results')
        #     save_image2(gen_imgs[0],out[0],imgsSIcolor[0], os.path.join(results_path, '{}_images'.format('proposed')),str(i))

        # Save first ten samples
        # if not saved_samples:
        #     saved_samples["imgs"] = imgscolor[:1].clone()
        #     saved_samples["masked"] = out.clone()
        #     saved_samples["lowres"] = imgsSIcolor[:1].clone()
        # if i>12000 :
        #     if i<12101:
        #        results_path = os.path.join('./outputs', 'results')
        #        save_image(gen_imgs[0],out[0],imgsSIcolor[0], os.path.join(results_path, '{}_images'.format('proposed')),str(i))
    
    # Validation
    print("validation ...")
    model.eval()
    val_loss = []
    val_mse = []
    val_mse2 = []
    # val_msssim = []
    val_bpp = []
    val_transmitted_bpp = []
    val_distortion = []
    with torch.no_grad():
        for ii, (imgscolor,imgsSIcolor, imgref, im2color, imSIcolor, im1color) in enumerate(iter(val_loader)):
            
            #if cuda:
    # imgsgray = imgsgray.type(Tensor)
            print("validation start!")
            imgref = imgref.type(Tensor)
            imgscolor = imgscolor.type(Tensor)
            imgsSIcolor = imgsSIcolor.type(Tensor)
            im2color = im2color.type(Tensor)
            imSIcolor = imSIcolor.type(Tensor)
    # masked_imgs = masked_imgs.type(Tensor)

# imgsgray = Variable(imgsgray)
            imgref = Variable(imgref)
            imgscolor = Variable(imgscolor)
            imgsSIcolor = Variable(imgsSIcolor)
            im2color = Variable(im2color)
            imSIcolor = Variable(imSIcolor)
            # img = input image, cor_img = side information/correlated image (designated y in the paper)
            #img, imgsi = data
            #img = img.cuda().float() if config['cuda'] else img.float()
            #cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()
            #imgsi = imgsi.cuda().float() if config['cuda'] else imgsi.float()
            
                #if (i%3)==0:
            out = model(imgscolor, imgsSIcolor)
                #else:
                 #   out = model(img, cor_img)
       
            bpp, transmitted_bpp = get_bpp(out, config)

            x_recon = out[0]
            
            out1 = Enhance(x_recon, imgsSIcolor)
            x_enhance = out1
            
            gen_imgs = generator(x_enhance, imSIcolor)
            
            mse_dist = mse(im2color, gen_imgs)
            mse_dist1 = mse(imgscolor, x_enhance)
            # msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                  # win_size=7)
            # msssim_db = -10 * np.log10(msssim)
            
            likelihoods = out[1]
            size_est = (-np.log(2) * x_recon.numel() / 1)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            
            #N,_,H, W = imgs_hr.size()
           # num_pixels = N * H * W
           # bpp_loss = (torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels))

            distortion = get_distortion(config, gen_imgs, im2color, im2color, mse)

            loss = distortion  # multiplied by (255 ** 2) for distortion scaling

            val_mse.append(mse_dist.item())
            val_mse2.append(mse_dist1.item())
            val_bpp.append(bpp.item())
            val_transmitted_bpp.append(transmitted_bpp.item())
            val_loss.append(loss.item())
            # val_msssim.append(msssim_db.item())
            val_distortion.append(distortion.item())
            print(10 * np.log10(1 / (sum(val_mse) / (len(val_mse)))))
            print(bpp)
            # print('bpp :', bpp)
            if ii<100:
                if config['save_image']:
                    results_path = os.path.join(config['save_output_path'], 'results')
                    save_image(gen_imgs,im2color,imSIcolor, os.path.join(results_path, '{}_images'.format(experiment_name)),
                                str(ii))
            if ii>101:
                break

    val_loss_to_track = 10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))
    # scheduler.step(val_loss_to_track)

    # Verbose
    tracking = ['Epoch {}:'.format(epoch + 1),
                    'Loss = {:.4f},'.format(val_loss_to_track),
                    'BPP = {:.8f},'.format(sum(val_bpp) / len(val_bpp)),
                    'Distortion = {:.4f},'.format(sum(val_distortion) / len(val_distortion)),
                    'Transmitted BPP = {:.8f},'.format(sum(val_transmitted_bpp) / len(val_transmitted_bpp)),
                    'PSNR gray = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))),
                    'PSNR color = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse2) / (len(val_mse2)))))]
                    # 'MS-SSIM = {:.4f}'.format(sum(val_msssim) / len(val_msssim))]
    print(" ".join(tracking))

    # Save weights
    if config['save_weights']:
        if min_val_loss is None or min_val_loss < val_loss_to_track:
            min_val_loss = val_loss_to_track
            save_path = os.path.join(weight_folder, 'colorization00026.pt')
            torch.save({
                "model_state_dict": generator.state_dict(),
                "optimizer_state_dict": optimizer_G.state_dict(),
            }, save_path)
            print('update_weights')
            
       
    # torch.save(generator.state_dict(),"outputs/weight/NAFNet.pth")
    # torch.save(discriminator.state_dict(), "outputs/weight/DiscriminatorCross.pth")
        # batches_done = epoch * len(train_loader) + i
        # if batches_done % optple_interval == 0:
        #     save_sample(saved_sample.sams)