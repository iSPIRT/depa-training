# Copyright (c) 2025 DEPA Foundation
#
# Licensed under CC0 1.0 Universal (https://creativecommons.org/publicdomain/zero/1.0/)
# 
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a 
# particular purpose and noninfringement. In no event shall the authors or copyright
# holders be liable for any claim, damages or other liability, whether in an action
# of contract, tort or otherwise, arising from, out of or in connection with the
# software or the use or other dealings in the software.
#
# For more information about this framework, please visit:
# https://depa.world/training/depa_training_framework/

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
import matplotlib.pyplot as plt # for saving sample predictions
import monai # for medical imaging loss functions
from tqdm import tqdm
import json

# Created for MRI segmentation scenario, 
class CustomDataset(Dataset):
    def __init__(self, data_dir, target_variable=None, transform=None, augment=False):
        self.transform = transform
        self.data_dir = data_dir
        self.target_variable = target_variable
        self.augment = augment
        
        # self.base_path = 'BraTS2020_Training_png'
        self.folder_pattern = 'BraTS20_Training_*'
        # self.image_prefix = 'BraTS20_Training_'
        self.patient_folders = glob(os.path.join(self.data_dir, self.folder_pattern))
        
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



def custom_loss_fn(outputs, labels):
    l1_loss = nn.L1Loss(reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

    return dice_loss(outputs, labels) + 2 * l1_loss(outputs, labels)

def dice_score(preds, targets, threshold=0.1, eps=1e-7):
    # If preds are logits, apply sigmoid
    if preds.dtype.is_floating_point:
        preds = torch.sigmoid(preds)

    # Calculate per batch
    batch_size = preds.size(0)
    dice_scores = []
    
    for i in range(batch_size):
        pred = preds[i].detach().squeeze().cpu().numpy() > threshold
        target = targets[i].detach().squeeze().cpu().numpy() > threshold
        
        # Calculate intersection and sum for dice score
        intersection = np.logical_and(target, pred)
        dice_score = (2 * np.sum(intersection)) / (np.sum(target) + np.sum(pred))
        dice_scores.append(dice_score)
        
    return np.mean(dice_scores)


def jaccard_index(preds, targets, threshold=0.1, eps=1e-7):
    # If preds are logits, apply sigmoid
    if preds.dtype.is_floating_point:
        preds = torch.sigmoid(preds)

    # Calculate per batch
    batch_size = preds.size(0)
    jaccard_scores = []
    
    for i in range(batch_size):
        pred = preds[i].detach().squeeze().cpu().numpy() > threshold
        target = targets[i].detach().squeeze().cpu().numpy() > threshold
        
        # Calculate intersection and union
        intersection = np.logical_and(target, pred)
        union = np.logical_or(target, pred)
        jaccard_score = np.sum(intersection) / np.sum(union)
        jaccard_scores.append(jaccard_score)

    return np.mean(jaccard_scores)



def custom_inference_fn(model, val_loader, device, config):
    """
    Custom inference function with additional metrics
    """
    save_path = config.get("paths", {}).get("sample_predictions_path")
    
    model.eval()
    dice_scores = []
    jaccard_indices = []
    total_loss = 0
    
    with torch.no_grad():
        # for [image, mask] in tqdm(val_loader, desc="Running inference"):
        for [image, mask] in val_loader:
            image = image.to(device)
            mask = mask.to(device)

            pred = model(image)

            loss = custom_loss_fn(pred, mask)
            total_loss += loss.item()

            dice = dice_score(pred, mask)
            jaccard = jaccard_index(pred, mask)

            dice_scores.append(dice)
            jaccard_indices.append(jaccard)

    # Calculate additional metrics
    avg_loss = total_loss / len(val_loader)

    # Calculate segmentation metrics
    avg_dice = np.mean(dice_scores)
    avg_jaccard = np.mean(jaccard_indices)

    metrics_dict = {
        'loss': avg_loss,
        'dice_score': avg_dice,
        'jaccard_index': avg_jaccard
    }
    # save as json
    metrics_fname = os.path.join(save_path, "validation_metrics.json")
    with open(metrics_fname, "w") as f:
        json.dump(metrics_dict, f)

    print(f"Validation Metrics: {metrics_dict}")

    # Save sample predictions
    i = 1
    with torch.no_grad():
        for image, mask in val_loader:
            image = image.to(device)
            mask = mask.to(device)
            pred = model(image)

            pred = pred[0].cpu().squeeze().numpy() > 0.1
            image = image[0].cpu().squeeze().numpy()
            mask = mask[0].cpu().squeeze().numpy() > 0.1
            plt.imsave(os.path.join(save_path, f"pred_{i}.png"), pred, cmap='gray')
            plt.imsave(os.path.join(save_path, f"image_{i}.png"), image, cmap='gray')
            plt.imsave(os.path.join(save_path, f"mask_{i}.png"), mask, cmap='gray')

            if i >= 5:
                break
            i += 1

    print(f"Sample predictions saved to {save_path}")

    return metrics_dict