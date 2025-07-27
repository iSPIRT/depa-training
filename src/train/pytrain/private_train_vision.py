import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms import ToTensor
import torchvision.transforms.functional as TF
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import opacus
from opacus import PrivacyEngine  # For differential privacy
from opacus.utils.batch_memory_manager import BatchMemoryManager  # For large batch sizes
from glob import glob
from tqdm import tqdm
import monai
import onnx
from onnx2pytorch import ConvertModel
from .task_base import TaskBase

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

class AnatomyUNet(nn.Module):
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


class PrivateTrainVision(TaskBase):
    """
    Args:
    config:training configuration 

    Methods:
    load_data: loads data from image folders as data loaders
    load_model: loads model
    load_optimizer: loads model optimizer and scheduler
    make_dprivate: wraps model, dataloader and optimizer with DP hooks
    loss_fn: loss function for training
    train: trains the model
    execute_model: mega function which includes all the above functions
    """

    def init(self, config):
        self.device = torch.device(config["DEVICE"])

        self.config = config
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.scheduler = None
        self.privacy_engine = None

    def load_data(self):
        dataset = CustomDataset(self.config["input_dataset_path"], augment=False)

        train_ratio = 1 - self.config["test_train_split"]
        n_samples = len(dataset)
        print(f"Total samples: {n_samples}")
        train_size = int(train_ratio*n_samples)

        self.config['delta'] = 1/train_size

        train_dataset = Subset(dataset, range(train_size))
        val_dataset = Subset(dataset, range(train_size, n_samples))
            
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        self.train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)#, collate_fn=lambda batch: [x for x in batch if x[0] is not None and x[1] is not None])
        self.val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)#, collate_fn=lambda batch: [x for x in batch if x[0] is not None and x[1] is not None])

    def join_datasets(self, dataset1, dataset2):
        # Join two datasets
        joined_dataset = dataset1 + dataset2
        return joined_dataset

    
    def load_model(self):
        ## Option 1 - load model state dict. Requires the model architecture to be defined.
        self.model = AnatomyUNet(in_ch=1, out_ch=1, base_ch=8, final_act='sigmoid').to(self.config['DEVICE'])
        self.model.load_state_dict(torch.load(self.config["saved_model_path"]))

        print("Model loaded")

        # ## Option 2 - load entire model using TorchScript
        # self.model = torch.jit.load(self.config["saved_model_path"])
        
        # # Option 3 - load ONNX model and convert to PyTorch
        # onnx_model = onnx.load(self.config["saved_model_path"])
        # model = ConvertModel(onnx_model, experimental=True)
        # # print(self.model)
        
        self.model = self.model.to(self.device)
    
    def load_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), self.config['LEARNING_RATE'])
        self.scheduler = CyclicLR(self.optimizer, base_lr=self.config['LEARNING_RATE'], max_lr=self.config['MAX_LR'], cycle_momentum=False)
    
    def make_dprivate(self):
        self.privacy_engine = PrivacyEngine() # secure_mode=True requires torchcsprng to be installed

        # for name, module in self.model.named_modules():
        #     if "Norm" in module.__class__.__name__:
        #         print(f"{name}: {module}")

                # raise ValueError("Norm layers are not supported in this model. Please remove them before training.")

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            epochs=self.config['total_epochs'],
            target_delta=self.config['delta'],  # Privacy budget
            target_epsilon=self.config['epsilon_threshold'],  # Probability of privacy breach
            max_grad_norm=self.config['max_grad_norm'], # threshold for clipping the norm of per-sample gradients
        )

    def loss_fn(self, pred, mask):
        l1_loss = nn.L1Loss(reduction='mean')
        mse_loss = nn.MSELoss(reduction='mean')
        dice_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
        return dice_loss(pred, mask) + 2 * l1_loss(pred, mask)


    def train(self):
        self.model = self.model.train()
        for epoch in range(self.config['total_epochs']):
            for [image, mask] in tqdm(self.train_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()

                # print(image.shape)
                
                pred = self.model(image)
                
                loss = self.loss_fn(pred, mask)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
            print(f"Epoch [{epoch+1}/{self.config['total_epochs']}], Loss: {loss.item():.4f}")

        # Extract the underlying model from GradSampleModule
        if isinstance(self.model, opacus.grad_sample.GradSampleModule):
            self.model = self.model._module

        output_path = self.config["trained_model_output_path"] + "trained_model.pth"
        # print("Writing training model to " + output_path)
        # torch.onnx.export(
        #     self.model.to('cpu'),
        #     self.val_loader[0][0].to('cpu'),                      # model input (or a tuple for multiple inputs)
        #     output_path,                # where to save the model
        #     input_names=["image"],            # input tensor names
        #     output_names=["mask"],          # output tensor names
        #     dynamic_axes={"image": {0: "batch_size"}, "mask": {0: "batch_size"}},  # optional
        #     export_params=True,               # store the trained parameter weights
        #     verbose=True,                  # print a human readable representation of the graph
        # )

        print("Writing training model to " + output_path)
        torch.save(self.model.state_dict(), output_path)

    def inference(self):
        self.model.eval()
        with torch.no_grad():
            i=1
            for [image, mask] in tqdm(self.val_loader):
                image = image.to(self.device)
                mask = mask.to(self.device)

                pred = self.model(image)
                loss = self.loss_fn(pred, mask)
                print(f"Validation Loss: {loss.item():.4f}")
                # Save the prediction and mask
                pred = pred[0].cpu().squeeze().numpy() > 0.1
                image = image[0].cpu().squeeze().numpy()
                mask = mask[0].cpu().squeeze().numpy() > 0.1
                plt.imsave(os.path.join(self.config["trained_model_output_path"], f"pred_{i}.png"), pred, cmap='gray')
                plt.imsave(os.path.join(self.config["trained_model_output_path"], f"image_{i}.png"), image, cmap='gray')
                plt.imsave(os.path.join(self.config["trained_model_output_path"], f"mask_{i}.png"), mask, cmap='gray')
                print(f"Sample predictions saved to {self.config['trained_model_output_path']}")
                i += 1
                if i > 5:
                    break

    def execute(self, config):
        try:
            # --- START OF TRAINING ---
            self.init(config)
            self.load_data()
            self.load_model()
            self.load_optimizer()
            self.make_dprivate()  # Differential privacy is not necessary for this task, but can be enabled if needed.
            self.train()
            print("Training complete!")
            # --- END OF TRAINING ---

            # for testing purposes, save some predictions
            self.inference()

        except Exception as e:
            print(f"An error occurred: {e}")
            raise e
