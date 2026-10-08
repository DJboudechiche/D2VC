import torch
import math
from models.balle2017 import entropy_model, gdn
from torch import nn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck
from models.attention_block import CrossAttention

lower_bound = entropy_model.lower_bound_fn.apply
#from compressai.zoo import bmshj2018_factorized


'''
The following model is based on the balle2018 model, which uses scale hyperpriors (z). 
'''
class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, start_from_relu=True, end_with_relu=False,
                 bottleneck=False):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=slope)
        if bottleneck:
            self.conv1 = nn.Conv2d(channel, channel // 2, 3, padding=1)
            self.conv2 = nn.Conv2d(channel // 2, channel, 3, padding=1)
        else:
            self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
            self.conv2 = nn.Conv2d(channel, channel, 3, padding=1)
        if start_from_relu:
            self.first_layer = self.leaky_relu
        else:
            self.first_layer = nn.Identity()
        if end_with_relu:
            self.last_layer = self.leaky_relu
        else:
            self.last_layer = nn.Identity()

    def forward(self, x):
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return x + out
    
    
class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.feature_conv = nn.Sequential(
            nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1),
            ResBlock(channel),
            ResBlock(channel),
        )
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.feature_conv(torch.cat((ctx, res), dim=1))
        recon = self.recon_conv(feature)
        return feature, recon
    

class MyAttention2(nn.Module):
    def __init__(self, M=192):
        super(MyAttention2, self).__init__()
        self.convv2 = nn.Conv2d(2*M, M, 1)
        self.conv = nn.Conv2d(M, M, 1, )
        self.m = nn.Softmax(dim=-1)
        self.act = nn.PReLU()
        self.gamma = 0.002
        
    def forward(self, f1, f2):
        
        x = torch.cat((f1, f2), 1)
        x = self.convv2(x)
        x = self.act(x)
        x = self.conv(x)
        x = self.act(x)
        out = x + f1
        # x1 = self.conv(f1)
        # x11 = self.act(x1)
        # y1 = self.conv(f2) 
        # # y11 = self.act(y1)
        # # a1 = x11 * y11
        # b1 = self.m(y1)       
        # out = b1 + f1 #the best
        # out = self.conv(out)
        return out

class HyperPriorDistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192, bound=0.11):
        super(HyperPriorDistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.ha_primary_image = HyperAnalysisTransform(num_filters)
        self.hs_primary_image = HyperSynthesisTransform(num_filters)

        self.ha_cor_image = HyperAnalysisTransform(num_filters)
        self.hs_cor_image = HyperSynthesisTransform(num_filters)

        self.entropy_bottleneck_sigma_x = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_sigma_y = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_common_info = entropy_model.EntropyBottleneck(num_filters, quantize=False)

        self.conditional_entropy_bottleneck_hx = ConditionalEntropyBottleneck()
        self.conditional_entropy_bottleneck_hy = ConditionalEntropyBottleneck(cor_input=True)

        self.deconv1 = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)

        return x

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)

        return x

    def forward(self, x, y):
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        # hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image

        z = self.ha_primary_image(abs(hx))
        z_tilde, z_likelihoods = self.entropy_bottleneck_sigma_x(z)
        sigma = self.hs_primary_image(z_tilde)
        sigma_lower_bounded = lower_bound(sigma, self.bound)
        
        # z_cor = self.ha_cor_image(abs(hy))
        # z_tilde_cor, z_likelihoods_cor = self.entropy_bottleneck_sigma_y(
        #     z_cor)
        # sigma_cor = self.hs_cor_image(z_tilde_cor)
        # sigma_cor_lower_bounded = lower_bound(sigma_cor, self.bound)

        hx_tilde, x_likelihoods = self.conditional_entropy_bottleneck_hx(hx, sigma_lower_bounded)
        # hy_tilde, y_likelihoods = self.conditional_entropy_bottleneck_hy(hy, sigma_cor_lower_bounded)

        w = self.encode_w(y)  # p(w|y), i.e. the "common variable"
        if self.training:
            w = w + math.sqrt(0.001) * torch.randn_like(w)  # Adding a small Gaussian noise improves stability of the training
        # _, w_likelihoods = self.entropy_bottleneck_common_info(w)

        x_tilde = self.decode(hx_tilde, w)
        # y_tilde = self.decode_cor(hy_tilde, w)

        return x_tilde, x_likelihoods, z_likelihoods


'''
This model is based on balle2017 model.
'''


class DistributedAutoEncoder(nn.Module):
    def __init__(self, num_filters=192,image_size=(64, 64), bound=0.11):
        super(DistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        
        self.deconv11 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.can1 = MyAttention2(M=num_filters)
        
        self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
                                  dim=128, num_patches=4, heads=4, dropout=0.1)
        self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
                                  dim=128, num_patches=4, heads=4, dropout=0.1)
        self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
                                  dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x
    
    # def decode(self, x, y):
    #     x = torch.cat((x, y), 1)
    #     #y = torch.cat((y, w), 1)

    #     x = self.deconv1(x)
    #     x = self.igdn1(x)
    #     y = self.deconv1_cor(y)
    #     y = self.igdn1_cor(y)
    #     # Applying cross attention between outputs of decoder's first layer
    #     x = self.ca1(x, y)

    #     x = self.deconv2(x)
    #     x = self.igdn2(x)
    #     y = self.deconv2_cor(y)
    #     y = self.igdn2_cor(y)
    #     # Applying cross attention between outputs of decoder's second layer
    #     x = self.ca2(x, y)

    #     x = self.deconv3(x)
    #     x = self.igdn3(x)
    #     y = self.deconv3_cor(y)
    #     y = self.igdn3_cor(y)
    #     # Applying cross attention between outputs of decoder's third layer
    #     x = self.ca3(x, y)

    #     x = self.deconv4(x)
        # y = self.deconv4_cor(y)

        # return x, y

    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x
    

    def forward(self, x, y):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        
        x1 = self.conv1_w(y)
        x2 = self.gdn1_w(x1)
        x2 = self.conv2_w(x2)
        x3 = self.gdn2_w(x2)
        x3 = self.conv3_w(x3)
        x4 = self.gdn3_w(x3)
        x4 = self.conv4_w(x4)
        
        
        if self.training:
           x = x + math.sqrt(0.001) * torch.randn_like(x)  # Adding small Gaussian noise improves the stability of training
        hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
        #hy = self.encode_cor(y)  # p(hy|y), i.e. the "private variable" of the correlated image
         
        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        #hy_tilde, y_likelihoods = self.entropy_bottleneck_hy(hy)
        #_, w_likelihoods = self.entropy_bottleneck(w)
        
        
        # x = torch.cat((hx_tilde, w), 1)
        out1 = torch.cat((hx_tilde, x4), 1)
        #y = torch.cat((y, w), 1)

        out1 = self.deconv1(out1)
        out1 = self.igdn1(out1)
      
        # Applying cross attention between outputs of decoder's first layer
        out2 = self.ca1(out1, x3)
        #out2 = torch.cat((out1, x3), 1)  # without cross attention

        out2 = self.deconv2(out2)
        out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer
        out3 = self.ca2(out2, x2)
        #out3 = torch.cat((out2, x2), 1)  # without cross attention

        out3 = self.deconv3(out3)
        out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        out = self.ca3(out3, x1)
        #out = torch.cat((out3, x1), 1)  # without cross attention

        x_tilde = self.deconv4(out)
        # x = self.deconv11(x)
        #x = self.deconv1(hx_tilde)        
        # out = self.can1(hx_tilde,w)
        # out = self.deconv11(hx_tilde,w)
        # x_tilde = self.decode(hx_tilde,w)
        
        # N,_,H, W = x.size()

        # num_pixels = N * H * W
        # bpp = torch.sum(torch.log(x_likelihoods)) / (-math.log(2) * num_pixels)
        # print(bpp)
        
        #y_tilde = self.decode_cor(hy_tilde, w)
        return x_tilde, x_likelihoods


class DistributedAutoEncoder2(nn.Module):
    def __init__(self, num_filters=192,image_size=(64, 64), bound=0.11):
        super(DistributedAutoEncoder2, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4 = gdn.GDN(num_filters)
        self.conv5 = nn.Conv2d(num_filters, 96, 5, stride=2, padding=2)
        self.conv6 = nn.Conv2d(96, num_filters, 5)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w96 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w96 = gdn.GDN(num_filters)
        self.conv2_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w96 = gdn.GDN(num_filters)
        self.conv3_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w96 = gdn.GDN(num_filters)
        self.conv4_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w128 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w128= gdn.GDN(num_filters)
        self.conv2_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w128 = gdn.GDN(num_filters)
        self.conv3_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w128 = gdn.GDN(num_filters)
        self.conv4_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)




        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx96 = entropy_model.EntropyBottleneck(96)
        self.entropy_bottleneck_hx128 = entropy_model.EntropyBottleneck(128)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        
        self.deconv5 = nn.ConvTranspose2d(96, num_filters, 5, stride=2, padding=6, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4 = gdn.GDN(num_filters, inverse=True)


        self.deconv1_96 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_96 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_96 = gdn.GDN(num_filters, inverse=True)


        
        self.deconv1_128 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_128 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_128 = gdn.GDN(num_filters, inverse=True)




        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.can1 = MyAttention2(M=num_filters)
        
        # self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x
    


    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x
    

    

    def forward(self, x, y, NL=96):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        
        #with torch.no_grad():
           
        #print(x4.shape)
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        hx = self.conv4(x)
        
        if self.training:
                x = x + math.sqrt(0.001) * torch.randn_like(x)  # Adding small Gaussian noise improves the stability of training
        #hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

           #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)

        if NL == 96:
            x1_96 = self.conv1_w96(y)
            x2_96 = self.gdn1_w96(x1_96)
            x2_96 = self.conv2_w96(x2_96)
            x3_96 = self.gdn2_w96(x2_96)
            x3_96 = self.conv3_w96(x3_96)
            x4_96 = self.gdn3_w96(x3_96)
            x4_96 = self.conv4_w96(x4_96)

            #hx2 = self.gdn4(hx)
            hx_96 = hx[:,:NL,:,:]
            hx_tilde, x_likelihoods = self.entropy_bottleneck_hx96(hx_96)
            x5_96 = x4_96[:,NL:,:,:]
            #print(hx.shape)
            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx96(hx)
            hx_tilde_96 = torch.cat((hx_tilde, x5_96), 1)
            #ÃƒÂ¢Ã¢â‚¬â€Ã‹Å“hx_tilde = self.deconv5(hx_tilde)
            #ÃƒÂ¢Ã‹Å“Ã‚Â»print(hx_tilde.shape)
            #hx_tilde2 = self.deconv5(hx_tilde2)
            #hx_tilde = self.igdn4(hx_tilde2)
            out1_96 = torch.cat((hx_tilde_96, x4_96), 1)
        #y = torch.cat((y, w), 1)

            out1_96 = self.deconv1_96(out1_96)
            out1_96 = self.igdn1_96(out1_96)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
            out2_96 = torch.cat((out1_96, x3_96), 1)  # without cross attention

            out2_96 = self.deconv2_96(out2_96)
            out2_96 = self.igdn2_96(out2_96)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
            out3_96 = torch.cat((out2_96, x2_96), 1)  # without cross attention

            out3_96 = self.deconv3_96(out3_96)
            out3_96 = self.igdn3_96(out3_96)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
            out_96 = torch.cat((out3_96, x1_96), 1)  # without cross attention
            x_tilde = self.deconv4_96(out_96)

        elif NL==128:
                x1_128 = self.conv1_w128(y)
                x2_128 = self.gdn1_w128(x1_128)
                x2_128 = self.conv2_w128(x2_128)
                x3_128 = self.gdn2_w128(x2_128)
                x3_128 = self.conv3_w128(x3_128)
                x4_128 = self.gdn3_w128(x3_128)
                x4_128 = self.conv4_w128(x4_128)
            
                #hx2 = self.gdn4(hx)
                hx_128 = hx[:,:NL,:,:]
                hx_tilde, x_likelihoods = self.entropy_bottleneck_hx128(hx_128)
                x5_128 = x4_128[:,NL:,:,:]
                #print(hx.shape)
                #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx128(hx)
                hx_tilde_128 = torch.cat((hx_tilde, x5_128), 1)
                #ÃƒÂ¢Ã¢â‚¬â€Ã‹Å“hx_tilde = self.deconv5(hx_tilde)
                #ÃƒÂ¢Ã‹Å“Ã‚Â»print(hx_tilde.shape)
                #hx_tilde2 = self.deconv5(hx_tilde2)
                #hx_tilde = self.igdn4(hx_tilde2)
                out1_128 = torch.cat((hx_tilde_128, x4_128), 1)
        #y = torch.cat((y, w), 1)

                out1_128 = self.deconv1_128(out1_128)
                out1_128 = self.igdn1_128(out1_128)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
                out2_128 = torch.cat((out1_128, x3_128), 1)  # without cross attention

                out2_128 = self.deconv2_128(out2_128)
                out2_128 = self.igdn2_128(out2_128)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
                out3_128 = torch.cat((out2_128, x2_128), 1)  # without cross attention

                out3_128 = self.deconv3_128(out3_128)
                out3_128 = self.igdn3_128(out3_128)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
                out_128 = torch.cat((out3_128, x1_128), 1)  # without cross attention

                x_tilde = self.deconv4_128(out_128)
        else:
            x1 = self.conv1_w(y)
            x2 = self.gdn1_w(x1)
            x2 = self.conv2_w(x2)
            x3 = self.gdn2_w(x2)
            x3 = self.conv3_w(x3)
            x4 = self.gdn3_w(x3)
            x4 = self.conv4_w(x4)

            hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
            
            out1 = torch.cat((hx_tilde, x4), 1)
        #y = torch.cat((y, w), 1)
            out1 = self.deconv1(out1)
            out1 = self.igdn1(out1)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
            out2 = torch.cat((out1, x3), 1)  # without cross attention

            out2 = self.deconv2(out2)
            out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
            out3 = torch.cat((out2, x2), 1)  # without cross attention

            out3 = self.deconv3(out3)
            out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
            out = torch.cat((out3, x1), 1)  # without cross attention

            x_tilde = self.deconv4(out)
            
        # x = torch.cat((hx_tilde, w), 1)
       

        return x_tilde, x_likelihoods


class DistributedAutoEncoder3(nn.Module):
    def __init__(self, num_filters=192, Quality=5, image_size=(64, 64), bound=0.11):
        super(DistributedAutoEncoder3, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4 = gdn.GDN(num_filters)
        self.conv5 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv6 = nn.Conv2d(96, num_filters, 5)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4_w = gdn.GDN(num_filters)
        self.conv5_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w96 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w96 = gdn.GDN(num_filters)
        self.conv2_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w96 = gdn.GDN(num_filters)
        self.conv3_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w96 = gdn.GDN(num_filters)
        self.conv4_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4_w96 = gdn.GDN(num_filters)
        self.conv5_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w128 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w128= gdn.GDN(num_filters)
        self.conv2_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w128 = gdn.GDN(num_filters)
        self.conv3_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w128 = gdn.GDN(num_filters)
        self.conv4_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4_w128 = gdn.GDN(num_filters)
        self.conv5_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)




        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx96 = entropy_model.EntropyBottleneck(96)
        self.entropy_bottleneck_hx128 = entropy_model.EntropyBottleneck(128)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        #Encoder = bmshj2018_factorized(quality=Quality, pretrained=True)
        #self.modelencoder = Encoder.g_a
        #self.modelencoderentropy = Encoder.entropy_bottleneck


        
        
        self.deconv1 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn4 = gdn.GDN(num_filters, inverse=True)
        self.deconv5 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)


        self.deconv1_96 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv5_96 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)


        
        self.deconv1_128 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv5_128 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)




        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv5_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)

        self.can1 = MyAttention2(M=num_filters)
        
        # self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x
    


    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x
    

    

    def forward(self, x, y, NL=96):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        hx = self.conv4(x)
        
        if self.training:
                x = x + math.sqrt(0.001) * torch.randn_like(x)  # Adding small Gaussian noise improves the stability of training
        #hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
        

      


        
        x1 = self.conv1_w(y)
        x2 = self.gdn1_w(x1)
        x2 = self.conv2_w(x2)
        x3 = self.gdn2_w(x2)
        x3 = self.conv3_w(x3)
        x4 = self.gdn3_w(x3)
        x4 = self.conv4_w(x4)
        #x5 = self.gdn4_w(x4)
        #x5 = self.conv5_w(x5)

            #print(x5.shape)
            #print(hx_tilde.shape)

            
        out1 = torch.cat((hx_tilde, x4), 1)
        #y = torch.cat((y, w), 1)
        out1 = self.deconv1(out1)
        out1 = self.igdn1(out1)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
        out2 = torch.cat((out1, x3), 1)  # without cross attention

        out2 = self.deconv2(out2)
        out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
        out3 = torch.cat((out2, x2), 1)  # without cross attention

        out3 = self.deconv3(out3)
        out3 = self.igdn3(out3)

        out4 = torch.cat((out3, x1), 1)  # without cross attention

        x_tilde = self.deconv5(out4)
        #out4 = self.igdn4(out4)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
        #out = torch.cat((out4, x1), 1)  # without cross attention

        #x_tilde = self.deconv5(out)
            
        # x = torch.cat((hx_tilde, w), 1)
       

        return x_tilde, x_likelihoods


class DistributedDecoder96(nn.Module):
    def __init__(self, num_filters=192,image_size=(64, 64), bound=0.11):
        super(DistributedDecoder96, self).__init__()



        self.conv1_w96 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w96 = gdn.GDN(num_filters)
        self.conv2_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w96 = gdn.GDN(num_filters)
        self.conv3_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w96 = gdn.GDN(num_filters)
        self.conv4_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        

        self.deconv1_96 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_96 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_96 = gdn.GDN(num_filters, inverse=True)

       
        # self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def forward(self, hx_tilde, y):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        
        x1_96 = self.conv1_w96(y)
        x2_96 = self.gdn1_w96(x1_96)
        x2_96 = self.conv2_w96(x2_96)
        x3_96 = self.gdn2_w96(x2_96)
        x3_96 = self.conv3_w96(x3_96)
        x4_96 = self.gdn3_w96(x3_96)
        x4_96 = self.conv4_w96(x4_96)

            #hx2 = self.gdn4(hx)
        hx_tilde_96 = hx_tilde[:,:96,:,:]
            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx96(hx_96)
        x5_96 = x4_96[:,96:,:,:]

        out1_96 = torch.cat((hx_tilde_96, x5_96), 1)
        #y = torch.cat((y, w), 1)

        out1_96 = self.deconv1_96(out1_96)
        out1_96 = self.igdn1_96(out1_96)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
        out2_96 = torch.cat((out1_96, x3_96), 1)  # without cross attention

        out2_96 = self.deconv2_96(out2_96)
        out2_96 = self.igdn2_96(out2_96)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
        out3_96 = torch.cat((out2_96, x2_96), 1)  # without cross attention

        out3_96 = self.deconv3_96(out3_96)
        out3_96 = self.igdn3_96(out3_96)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
        out_96 = torch.cat((out3_96, x1_96), 1)  # without cross attention
        x_tilde = self.deconv4_96(out_96)

       
        return x_tilde

class DistributedDecoder192(nn.Module):
    def __init__(self, num_filters=192,image_size=(64, 64), bound=0.11):
        super(DistributedDecoder192, self).__init__()

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
       
        self.deconv5 = nn.ConvTranspose2d(96, num_filters, 5, stride=2, padding=6, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4 = gdn.GDN(num_filters, inverse=True)


        

        self.bound = bound   

    def forward(self, hx_tilde, y):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        x1 = self.conv1_w(y)
        x2 = self.gdn1_w(x1)
        x2 = self.conv2_w(x2)
        x3 = self.gdn2_w(x2)
        x3 = self.conv3_w(x3)
        x4 = self.gdn3_w(x3)
        x4 = self.conv4_w(x4)

            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
            
        out1 = torch.cat((hx_tilde, x4), 1)
        #y = torch.cat((y, w), 1)
        out1 = self.deconv1(out1)
        out1 = self.igdn1(out1)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
        out2 = torch.cat((out1, x3), 1)  # without cross attention

        out2 = self.deconv2(out2)
        out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
        out3 = torch.cat((out2, x2), 1)  # without cross attention

        out3 = self.deconv3(out3)
        out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
        out = torch.cat((out3, x1), 1)  # without cross attention

        x_tilde = self.deconv4(out)
            
        # x = torch.cat((hx_tilde, w), 1)
       

        return x_tilde


class DistributedAutoEncoder4(nn.Module):
    def __init__(self, num_filters=192,image_size=(64, 64), bound=0.11):
        super(DistributedAutoEncoder4, self).__init__()
        
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn4 = gdn.GDN(num_filters)
        self.conv5 = nn.Conv2d(num_filters, 96, 5, stride=2, padding=2)
        self.conv6 = nn.Conv2d(96, num_filters, 5)

        self.conv1_cor = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w = gdn.GDN(num_filters)
        self.conv2_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w = gdn.GDN(num_filters)
        self.conv3_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w = gdn.GDN(num_filters)
        self.conv4_w = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w96 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w96 = gdn.GDN(num_filters)
        self.conv2_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w96 = gdn.GDN(num_filters)
        self.conv3_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w96 = gdn.GDN(num_filters)
        self.conv4_w96 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)


        self.conv1_w128 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1_w128= gdn.GDN(num_filters)
        self.conv2_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_w128 = gdn.GDN(num_filters)
        self.conv3_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_w128 = gdn.GDN(num_filters)
        self.conv4_w128 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)




        self.entropy_bottleneck = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        self.entropy_bottleneck_hx96 = entropy_model.EntropyBottleneck(96)
        self.entropy_bottleneck_hx128 = entropy_model.EntropyBottleneck(128)
        self.entropy_bottleneck_hx = entropy_model.EntropyBottleneck(num_filters)
        self.entropy_bottleneck_hy = entropy_model.EntropyBottleneck(num_filters, quantize=False)
        
        self.deconv5 = nn.ConvTranspose2d(96, num_filters, 5, stride=2, padding=6, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1 = gdn.GDN(num_filters, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4 = gdn.GDN(num_filters, inverse=True)


        self.deconv1_96 = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_96 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_96 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_96 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_96 = gdn.GDN(num_filters, inverse=True)


        
        self.deconv1_128 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv2_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv3_128 = nn.ConvTranspose2d(2*num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_128 = gdn.GDN(num_filters, inverse=True)
        self.deconv4_128 = nn.ConvTranspose2d(2*num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_128 = gdn.GDN(num_filters, inverse=True)




        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 3, 5, stride=2, padding=2, output_padding=1)
        self.can1 = MyAttention2(M=num_filters)
        
        # self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def encode(self, x):
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conv4(x)
        return x

    def encode_cor(self, x):
        x = self.conv1_cor(x)
        x = self.gdn1_cor(x)
        x = self.conv2_cor(x)
        x = self.gdn2_cor(x)
        x = self.conv3_cor(x)
        x = self.gdn3_cor(x)
        x = self.conv4_cor(x)
        return x

    def encode_w(self, x):
        x = self.conv1_w(x)
        x = self.gdn1_w(x)
        x = self.conv2_w(x)
        x = self.gdn2_w(x)
        x = self.conv3_w(x)
        x = self.gdn3_w(x)
        x = self.conv4_w(x)
        return x

    def decode(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1(x)
        x = self.igdn1(x)
        x = self.deconv2(x)
        x = self.igdn2(x)
        x = self.deconv3(x)
        x = self.igdn3(x)
        x = self.deconv4(x)
        return x
    


    def decode_cor(self, x, w):
        x = torch.cat((x, w), 1)
        x = self.deconv1_cor(x)
        x = self.igdn1_cor(x)
        x = self.deconv2_cor(x)
        x = self.igdn2_cor(x)
        x = self.deconv3_cor(x)
        x = self.igdn3_cor(x)
        x = self.deconv4_cor(x)
        return x
    

    

    def forward(self, x, y, NL=96):
        # w = self.encode_w(y)  # p(w|y), i.e. the "common variable "
        
        #with torch.no_grad():
           
        #print(x4.shape)
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conv3(x)
        x = self.gdn3(x)
        hx = self.conv4(x)
        
        if self.training:
                x = x + math.sqrt(0.001) * torch.randn_like(x)  # Adding small Gaussian noise improves the stability of training
        #hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image

        hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)

        if NL == 96:
            x1_96 = self.conv1_w96(y)
            x2_96 = self.gdn1_w96(x1_96)
            x2_96 = self.conv2_w96(x2_96)
            x3_96 = self.gdn2_w96(x2_96)
            x3_96 = self.conv3_w96(x3_96)
            x4_96 = self.gdn3_w96(x3_96)
            x4_96 = self.conv4_w96(x4_96)

            #hx2 = self.gdn4(hx)
            hx_tilde_96 = hx_tilde[:,:NL,:,:]
            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx96(hx_96)
            x5_96 = x4_96[:,NL:,:,:]
            #print(hx.shape)
            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx96(hx)
            #hx_tilde_96 = torch.cat((hx_tilde, x5_96), 1)
            #â—˜hx_tilde = self.deconv5(hx_tilde)
            #â˜»print(hx_tilde.shape)
            #hx_tilde2 = self.deconv5(hx_tilde2)
            #hx_tilde = self.igdn4(hx_tilde2)
            out1_96 = torch.cat((hx_tilde_96, x5_96), 1)
        #y = torch.cat((y, w), 1)

            out1_96 = self.deconv1_96(out1_96)
            out1_96 = self.igdn1_96(out1_96)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
            out2_96 = torch.cat((out1_96, x3_96), 1)  # without cross attention

            out2_96 = self.deconv2_96(out2_96)
            out2_96 = self.igdn2_96(out2_96)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
            out3_96 = torch.cat((out2_96, x2_96), 1)  # without cross attention

            out3_96 = self.deconv3_96(out3_96)
            out3_96 = self.igdn3_96(out3_96)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
            out_96 = torch.cat((out3_96, x1_96), 1)  # without cross attention
            x_tilde = self.deconv4_96(out_96)

        elif NL==128:
                x1_128 = self.conv1_w128(y)
                x2_128 = self.gdn1_w128(x1_128)
                x2_128 = self.conv2_w128(x2_128)
                x3_128 = self.gdn2_w128(x2_128)
                x3_128 = self.conv3_w128(x3_128)
                x4_128 = self.gdn3_w128(x3_128)
                x4_128 = self.conv4_w128(x4_128)
            
                #hx2 = self.gdn4(hx)
                hx_128 = hx[:,:NL,:,:]
                hx_tilde, x_likelihoods = self.entropy_bottleneck_hx128(hx_128)
                x5_128 = x4_128[:,NL:,:,:]
                #print(hx.shape)
                #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx128(hx)
                hx_tilde_128 = torch.cat((hx_tilde, x5_128), 1)
                #â—˜hx_tilde = self.deconv5(hx_tilde)
                #â˜»print(hx_tilde.shape)
                #hx_tilde2 = self.deconv5(hx_tilde2)
                #hx_tilde = self.igdn4(hx_tilde2)
                out1_128 = torch.cat((hx_tilde_128, x4_128), 1)
        #y = torch.cat((y, w), 1)

                out1_128 = self.deconv1_128(out1_128)
                out1_128 = self.igdn1_128(out1_128)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
                out2_128 = torch.cat((out1_128, x3_128), 1)  # without cross attention

                out2_128 = self.deconv2_128(out2_128)
                out2_128 = self.igdn2_128(out2_128)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
                out3_128 = torch.cat((out2_128, x2_128), 1)  # without cross attention

                out3_128 = self.deconv3_128(out3_128)
                out3_128 = self.igdn3_128(out3_128)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
                out_128 = torch.cat((out3_128, x1_128), 1)  # without cross attention

                x_tilde = self.deconv4_128(out_128)
        else:
            x1 = self.conv1_w(y)
            x2 = self.gdn1_w(x1)
            x2 = self.conv2_w(x2)
            x3 = self.gdn2_w(x2)
            x3 = self.conv3_w(x3)
            x4 = self.gdn3_w(x3)
            x4 = self.conv4_w(x4)

            #hx_tilde, x_likelihoods = self.entropy_bottleneck_hx(hx)
            
            out1 = torch.cat((hx_tilde, x4), 1)
        #y = torch.cat((y, w), 1)
            out1 = self.deconv1(out1)
            out1 = self.igdn1(out1)
      
        # Applying cross attention between outputs of decoder's first layer
        #out2 = self.ca1(out1, x3)
            out2 = torch.cat((out1, x3), 1)  # without cross attention

            out2 = self.deconv2(out2)
            out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer
        #out3 = self.ca2(out2, x2)
            out3 = torch.cat((out2, x2), 1)  # without cross attention

            out3 = self.deconv3(out3)
            out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1)
            out = torch.cat((out3, x1), 1)  # without cross attention

            x_tilde = self.deconv4(out)
            
        # x = torch.cat((hx_tilde, w), 1)
       

        return x_tilde, x_likelihoods, hx_tilde

if __name__ == '__main__':

    
    net = DistributedEncoder().cuda()
    print(net(torch.randn(1, 3, 128, 128).cuda())[0].shape)



