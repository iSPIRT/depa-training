import os
from glob import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


# Created for MRI segmentation scenario, 
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=True):
        self.transform = transform
        self.root_dir = root_dir
        self.augment = augment
        
        # self.base_path = 'BraTS2020_Training_png'
        self.folder_pattern = 'BraTS20_Training_*'
        # self.image_prefix = 'BraTS20_Training_'
        self.patient_folders = glob(os.path.join(self.root_dir, self.folder_pattern))
        
        # create pairs of images, masks
        self.samples = []
        for patient_folder in self.patient_folders:
            patient_id = os.path.basename(patient_folder)
            flair_files = sorted(glob(os.path.join(patient_folder, f"{patient_id}_flair*.png")))
            
            for flair_file in flair_files:
                # slice number from flair filename
                slice_name = os.path.basename(flair_file)
                slice_number = slice_name.replace(f"{patient_id}_flair", "").replace(".png", "")
                
                # corresponding segmentation mask
                mask_file = os.path.join(patient_folder, f"{patient_id}_seg{slice_number}.png")
                
                # mask exists?
                if os.path.exists(mask_file):
                    m = cv2.imread(mask_file)
                    if not np.all(m==0):
                        self.samples.append((flair_file, mask_file))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        i = cv2.imread(img_path)
        m = cv2.imread(mask_path)

        image = Image.fromarray(i).convert('L')
        mask = Image.fromarray(m).convert('L')
        
        # convert tensor
        image = ToTensor()(image)
        mask = ToTensor()(mask)

        # binarize mask (any non-zero value becomes 1)
        mask = (mask > 0).float()
        
        # Apply additional transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, mask



# Model architecture and components for Anatomy UNet
class ConvBlock2d(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, 1, 1),
            # nn.InstanceNorm2d(mid_ch),
            nn.GroupNorm(1, mid_ch),
            nn.LeakyReLU(0.1),
            nn.Conv2d(mid_ch, out_ch, 3, 1, 1),
            # nn.InstanceNorm2d(out_ch),
            nn.GroupNorm(1, out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor):
        return self.conv(in_tensor)


class Upsample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = in_ch // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            # nn.InstanceNorm2d(out_ch),
            nn.GroupNorm(1, out_ch),
            nn.LeakyReLU(0.1)
        )

    def forward(self, in_tensor, encoded_feature):
        up_sampled_tensor = F.interpolate(in_tensor, size=None, scale_factor=2.0, mode='bilinear', align_corners=False)
        up_sampled_tensor = self.conv(up_sampled_tensor)
        return torch.cat([encoded_feature, up_sampled_tensor], dim=1)

class Base_Model(nn.Module):
    def __init__(self, in_ch, out_ch, conditional_ch=0, num_lvs=4, base_ch=16, final_act='noact'):
        super().__init__()
        self.final_act = final_act
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.down_convs = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for lv in range(num_lvs):
            ch = base_ch * (2 ** lv)
            self.down_convs.append(ConvBlock2d(ch + conditional_ch, ch * 2, ch * 2))
            self.down_samples.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.up_samples.append(Upsample(ch * 4))
            self.up_convs.append(ConvBlock2d(ch * 4, ch * 2, ch * 2))
        bottleneck_ch = base_ch * (2 ** num_lvs)
        self.bottleneck_conv = ConvBlock2d(bottleneck_ch, bottleneck_ch * 2, bottleneck_ch * 2)
        self.out_conv = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch, 3, 1, 1),
                                      nn.LeakyReLU(0.1),
                                      nn.Conv2d(base_ch, out_ch, 3, 1, 1))

    def forward(self, in_tensor, condition=None):
        encoded_features = []
        x = self.in_conv(in_tensor)
        for down_conv, down_sample in zip(self.down_convs, self.down_samples):
            if condition is not None:
                feature_dim = x.shape[-1]
                down_conv_out = down_conv(torch.cat([x, condition.repeat(1, 1, feature_dim, feature_dim)], dim=1))
            else:
                down_conv_out = down_conv(x)
            x = down_sample(down_conv_out)
            encoded_features.append(down_conv_out)
        x = self.bottleneck_conv(x)
        for encoded_feature, up_conv, up_sample in zip(reversed(encoded_features),
                                                       reversed(self.up_convs),
                                                       reversed(self.up_samples)):
            x = up_sample(x, encoded_feature)
            x = up_conv(x)
        x = self.out_conv(x)
        if self.final_act == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.final_act == "relu":
            x = torch.relu(x)
        elif self.final_act == 'tanh':
            x = torch.tanh(x)
        else:
            x = x
        return x