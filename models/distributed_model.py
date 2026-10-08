import torch
import math
from models.balle2017 import entropy_model, gdn
from torch import nn
from models.balle2018.hypertransforms import HyperAnalysisTransform, HyperSynthesisTransform
from models.balle2018.conditional_entropy_model import ConditionalEntropyBottleneck
from models.attention_block import CrossAttention

lower_bound = entropy_model.lower_bound_fn.apply


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
    def __init__(self, num_filters=192,image_size=(144, 176), bound=0.11):
        super(DistributedAutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(1, num_filters, 5, stride=2, padding=2)
        self.gdn1_cor = gdn.GDN(num_filters)
        self.conv2_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2_cor = gdn.GDN(num_filters)
        self.conv3_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3_cor = gdn.GDN(num_filters)
        self.conv4_cor = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_w = nn.Conv2d(1, num_filters, 5, stride=2, padding=2)
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
        self.deconv2 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1) # change
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1) # change
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2* num_filters, 1, 5, stride=2, padding=2, output_padding=1) # change

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 1, 5, stride=2, padding=2, output_padding=1)
        self.can1 = MyAttention2(M=num_filters)
        
        self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        
        

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
        #out2 = self.ca1(out1, x3) # proposed
        out2 = torch.cat((out1, x3), 1)  # without cross attention

        out2 = self.deconv2(out2) # out2
        out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer

        #out3 = self.ca2(out2, x2) # proposed
        out3 = torch.cat((out2, x2), 1)  # without cross attention

        out3 = self.deconv3(out3) # out3
        out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1) # proposed
        out = torch.cat((out3, x1), 1)  # without cross attention

        x_tilde = self.deconv4(out) #out
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
        return x_tilde, x_likelihoods #, hx_tilde



# Example FiLM layer
class FiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)
        self.act = nn.GELU()

    def forward(self, x, cond):
        # x: (B, C, H, W), cond: (B, cond_dim)
        gamma = self.gamma(cond).unsqueeze(2).unsqueeze(3)
        #print(gamma.shape)
        #gamma = gamma.permute(0,3,1,2)
        #beta = self.beta(cond).unsqueeze(2).unsqueeze(3)
        # print(x.shape)
        # print(gamma.shape)
        return self.act(gamma * x) #+ beta



class DistributedAutoEncoderconditional(nn.Module):
    def __init__(self, num_filters=192,image_size=(144, 176), bound=0.11):
        super(DistributedAutoEncoderconditional, self).__init__()
        
        self.conditional = FiLM(num_filters, cond_dim=1)
        
        self.conv1 = nn.Conv2d(3, num_filters, 5, stride=2, padding=2)
        self.gdn1 = gdn.GDN(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn2 = gdn.GDN(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)
        self.gdn3 = gdn.GDN(num_filters)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 5, stride=2, padding=2)

        self.conv1_cor = nn.Conv2d(1, num_filters, 5, stride=2, padding=2)
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
        self.deconv2 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1) # change
        self.igdn2 = gdn.GDN(num_filters, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(2* num_filters, num_filters, 5, stride=2, padding=2, output_padding=1) # change
        self.igdn3 = gdn.GDN(num_filters, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(2* num_filters, 3, 5, stride=2, padding=2, output_padding=1) # change

        self.deconv1_cor = nn.ConvTranspose2d(2 * num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn1_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv2_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn2_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv3_cor = nn.ConvTranspose2d(num_filters, num_filters, 5, stride=2, padding=2, output_padding=1)
        self.igdn3_cor = gdn.GDN(num_filters, inverse=True)
        self.deconv4_cor = nn.ConvTranspose2d(num_filters, 1, 5, stride=2, padding=2, output_padding=1)
        self.can1 = MyAttention2(M=num_filters)
        
        self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
                                  dim=128, num_patches=2, dropout=0.1)
        
        

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
    

    def forward(self, x, y, cond):
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
        x = self.conv1(x)
        x = self.gdn1(x)
        x = self.conditional(x,cond)
        x = self.conv2(x)
        x = self.gdn2(x)
        x = self.conditional(x,cond)
        x = self.conv3(x)
        x = self.gdn3(x)
        x = self.conditional(x,cond)
        hx = self.conv4(x)
        
        # hx = self.encode(x)  # p(hx|x), i.e. the "private variable" of the primary image
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
        #out2 = self.ca1(out1, x3) # proposed
        out2 = torch.cat((out1, x3), 1)  # without cross attention

        out2 = self.deconv2(out2) # out2
        out2 = self.igdn2(out2)

        # Applying cross attention between outputs of decoder's second layer

        #out3 = self.ca2(out2, x2) # proposed
        out3 = torch.cat((out2, x2), 1)  # without cross attention

        out3 = self.deconv3(out3) # out3
        out3 = self.igdn3(out3)
      
        # Applying cross attention between outputs of decoder's third layer
        #out = self.ca3(out3, x1) # proposed
        out = torch.cat((out3, x1), 1)  # without cross attention

        x_tilde = self.deconv4(out) #out
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
    
class DistributedDecoder96Y(nn.Module):
    def __init__(self, num_filters=192,image_size=(144, 176), bound=0.11):
        super(DistributedDecoder96Y, self).__init__()



        self.conv1_w96 = nn.Conv2d(1, num_filters, 5, stride=2, padding=2)
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
        self.deconv4_96 = nn.ConvTranspose2d(2*num_filters, 1, 5, stride=2, padding=2, output_padding=1)
        self.igdn4_96 = gdn.GDN(num_filters, inverse=True)

       
        # self.ca1 = CrossAttention(input_size=(image_size[0] // 8, image_size[1] // 8), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca2 = CrossAttention(input_size=(image_size[0] // 4, image_size[1] // 4), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        # self.ca3 = CrossAttention(input_size=(image_size[0] // 2, image_size[1] // 2), num_filters=32,
        #                           dim=128, num_patches=4, heads=4, dropout=0.1)
        

        self.bound = bound

    def forward(self, hx_tilde, y, x_likelihoods):
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

       
        return x_tilde, x_likelihoods

if __name__ == '__main__':
    # net = HyperPriorDistributedAutoEncoder().cuda()
    # print(net(torch.randn(1, 3, 256, 256).cuda(), torch.randn(1, 3, 256, 256).cuda())[0].shape)
    net = DistributedAutoEncoder().cuda()
    net2 =  DistributedDecoder96Y().cuda()
    # print(net(torch.randn(1, 1, 144, 176).cuda(), torch.randn(1, 1, 144, 176).cuda(), torch.randn(1,1).cuda())[0].shape)
    a,b,c = net(torch.randn(1, 3, 64, 64).cuda(), torch.randn(1, 3, 64, 64).cuda())
    d = net2(c,torch.randn(1, 3, 64, 64).cuda())
    print(d.shape)

