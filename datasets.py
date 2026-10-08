import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
#import cv2
import numpy 
import numpy as np
import lpips
import torch

loss_fn = lpips.LPIPS(net='alex').cuda()


class DBreader_Vimeo90k(Dataset):
    def __init__(
        self,
        root,
        path_list,
        random_crop=None,
        resize=None,
        augment_s=True,
        augment_t=True,
    ):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, "sequences", im_path)

        transform_list = []
        transform_list1 = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()] #transforms.Grayscale(num_output_channels=1),
        transform_list1 += [transforms.ToTensor()]
        #transform_list1 += [transforms.Grayscale(num_output_channels=1),transforms.ToTensor()] #transforms.Grayscale(num_output_channels=1),
        self.transform = transforms.Compose(transform_list)
        self.transform1 = transforms.Compose(transform_list1 )

        rawFrame0 = Image.open(os.path.join(abs_im_path, "imgc1.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im2.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "imgside.png")) #imgside
        rawFrame3 = Image.open(os.path.join(abs_im_path, "im3.png"))
        rawFrame4 = Image.open(os.path.join(abs_im_path, "imgside.png"))
        
        

        

        

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                rawFrame1, output_size=[self.random_crop,self.random_crop]
            )
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            rawFrame3 = TF.crop(rawFrame3, i, j, h, w)
            rawFrame4 = TF.crop(rawFrame4, i, j, h, w)


        
        
        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
                rawFrame3 = TF.hflip(rawFrame3)
                #rawFrame4 = TF.hflip(rawFrame4)
                
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
                rawFrame3 = TF.vflip(rawFrame3)
                #rawFrame4 = TF.vflip(rawFrame4)

        #rawFrame0cv = numpy.array(rawFrame0)
        #result, rawFrame0cv = cv2.imencode('.jpg', np.float32(rawFrame0cv), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        #rawFrame0cv = cv2.imdecode(rawFrame0cv, 1)
              
        #rawFrame1cv = numpy.array(rawFrame1)
              #Presult, rawFrame1cv = cv2.imencode('.jpg', np.float32(rawFrame1cv), [int(cv2.IMWRITE_JPEG_QUALITY), 20])
              #rawFrame1cv = cv2.imdecode(rawFrame1cv, 1)
              
        #rawFrame2cv = numpy.array(rawFrame2)
        #result, rawFrame2cv = cv2.imencode('.jpg', np.float32(rawFrame2cv), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        #rawFrame2cv = cv2.imdecode(rawFrame2cv, 1)  
        
        frame0 = self.transform1(rawFrame0)
        frame1 = self.transform1(rawFrame1)  #GT/2 
        frame2 = self.transform1(rawFrame2) #GT
        frame3 = self.transform1(rawFrame3)  #SI/2
        #frame4 = self.transform1(rawFrame2)  #SI
        #frame5 = self.transform(rawFrame0)  #SI
        
        
        # im1 = (frame0.cuda() - 0.5) * 2
        # im2 = (frame2.cuda() - 0.5) * 2
        # SIim = (frame3.cuda() - 0.5) * 2

                        # Compute LPIPS
        # with torch.no_grad():
        #       targets1 = loss_fn(im2, SIim)
        #       targets2 = loss_fn(im2, im1)

        # if self.augment_t:
        #     if random.randint(0, 1):
        #         return frame0, frame1, frame2, frame3, frame4 #targets1, targets2
        #     else:
        #         return  frame0, frame1, frame2, frame3, frame4 #targets1, targets2
        # else:
        #     return  frame0, frame1, frame2, frame3, frame4 #targets1, targets2
        
        if self.augment_t:
            if random.randint(0, 1):
                return frame1, frame2, frame0, frame3#,frame0 #targets1, targets2
            else:
                return  frame1, frame2, frame0, frame3#,frame0 #targets1, targets2
        else:
            return  frame1, frame2, frame0, frame3#,frame0#targets1, targets2
        
class DBreader_Vimeo90k2(Dataset):
    def __init__(
        self,
        root,
        path_list,
        random_crop=None,
        resize=None,
        augment_s=True,
        augment_t=True,
    ):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, "sequences", im_path)

        transform_list = []
        transform_list1 = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        transform_list += [transforms.ToTensor()] #transforms.Grayscale(num_output_channels=1),
        transform_list1 += [transforms.Grayscale(num_output_channels=1),transforms.ToTensor()] #transforms.Grayscale(num_output_channels=1),
        self.transform = transforms.Compose(transform_list)
        self.transform1 = transforms.Compose(transform_list1)

        rawFrame0 = Image.open(os.path.join(abs_im_path, "im2.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im1.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "imgside.png")) #imgside
        rawFrame3 = Image.open(os.path.join(abs_im_path, "imgc1.png"))
        

        

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                rawFrame1, output_size=[self.random_crop,self.random_crop]
            )
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)
            rawFrame3 = TF.crop(rawFrame3, i, j, h, w)

        
        
        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
                rawFrame3 = TF.hflip(rawFrame3)
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)
                rawFrame3 = TF.vflip(rawFrame3)

        #rawFrame0cv = numpy.array(rawFrame0)
        #result, rawFrame0cv = cv2.imencode('.jpg', np.float32(rawFrame0cv), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        #rawFrame0cv = cv2.imdecode(rawFrame0cv, 1)
              
        #rawFrame1cv = numpy.array(rawFrame1)
              #Presult, rawFrame1cv = cv2.imencode('.jpg', np.float32(rawFrame1cv), [int(cv2.IMWRITE_JPEG_QUALITY), 20])
              #rawFrame1cv = cv2.imdecode(rawFrame1cv, 1)
              
        #rawFrame2cv = numpy.array(rawFrame2)
        #result, rawFrame2cv = cv2.imencode('.jpg', np.float32(rawFrame2cv), [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        #rawFrame2cv = cv2.imdecode(rawFrame2cv, 1)  
        
        frame0 = self.transform(rawFrame0)
        frame1 = self.transform1(rawFrame1)
        frame2 = self.transform1(rawFrame2)
        frame3 = self.transform1(rawFrame3)

        if self.augment_t:
            if random.randint(0, 1):
                return frame0,frame2, frame3, frame1 
            else:
                return  frame0,frame2, frame3, frame1 
        else:
            return  frame0,frame2, frame3, frame1 

class DBreader_Vimeo90kY(Dataset):
    def __init__(
        self,
        root,
        path_list,
        random_crop=None,
        resize=None,
        augment_s=True,
        augment_t=True,
    ):
        self.root = root
        self.path_list = path_list

        self.random_crop = random_crop
        self.resize = resize
        self.augment_s = augment_s
        self.augment_t = augment_t

    def __getitem__(self, index):
        path = self.path_list[index]
        return self.Vimeo90K_loader(path)

    def __len__(self):
        return len(self.path_list)

    def Vimeo90K_loader(self, im_path):
        abs_im_path = os.path.join(self.root, "sequences", im_path)

        transform_list = []
        if self.resize is not None:
            transform_list += [transforms.Resize(self.resize)]
        #transform_list += [transforms.Grayscale(num_output_channels=1)]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(transform_list)

        rawFrame0 = Image.open(os.path.join(abs_im_path, "im1.png"))
        rawFrame1 = Image.open(os.path.join(abs_im_path, "im2.png"))
        rawFrame2 = Image.open(os.path.join(abs_im_path, "im3.png"))

        if self.random_crop is not None:
            i, j, h, w = transforms.RandomCrop.get_params(
                rawFrame1, output_size=self.random_crop
            )
            rawFrame0 = TF.crop(rawFrame0, i, j, h, w)
            rawFrame1 = TF.crop(rawFrame1, i, j, h, w)
            rawFrame2 = TF.crop(rawFrame2, i, j, h, w)

        if self.augment_s:
            if random.randint(0, 1):
                rawFrame0 = TF.hflip(rawFrame0)
                rawFrame1 = TF.hflip(rawFrame1)
                rawFrame2 = TF.hflip(rawFrame2)
            if random.randint(0, 1):
                rawFrame0 = TF.vflip(rawFrame0)
                rawFrame1 = TF.vflip(rawFrame1)
                rawFrame2 = TF.vflip(rawFrame2)

        frame0 = self.transform(rawFrame0)
        frame1 = self.transform(rawFrame1)
        frame2 = self.transform(rawFrame2)

        if self.augment_t:
            if random.randint(0, 1):
                return frame2, frame1, frame0
            else:
                return frame0, frame1, frame2
        else:
            return frame0, frame1, frame2



def make_dataset(root, list_file, num_training_samples=-1):
    raw_im_list = open(os.path.join(root, list_file)).read().splitlines()
    raw_im_list = raw_im_list[:-1]  # the last line is invalid in test set
    assert len(raw_im_list) > 0
    random.shuffle(raw_im_list)
    if num_training_samples <= 0:
        return raw_im_list
    else:
        return raw_im_list[-num_training_samples:]


def Vimeo90K_interp(root, num_training_samples=-1):
    train_list = make_dataset(root, "tri_trainlist.txt", num_training_samples)
    val_list = make_dataset(root, "tri_testlist.txt")
    train_dataset = DBreader_Vimeo90k(root, train_list,random_crop=128, resize=None)
    val_dataset = DBreader_Vimeo90k(root, val_list, random_crop=128, resize=None)
    return train_dataset , val_dataset