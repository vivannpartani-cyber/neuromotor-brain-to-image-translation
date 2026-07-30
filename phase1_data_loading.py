import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from nilearn import datasets
from PIL import Image
import torchvision.transforms as transforms

class RealFMRIDataset(Dataset):
    def __init__(self, data_split="train", max_runs=10, img_save_dir="data/miyawaki_images"):
        """
        Loads the Miyawaki 2008 dataset: Real V1-V4 fMRI responses to 10x10 images.
        """
        print(f"Loading Miyawaki 2008 {data_split} dataset...")
        self.miyawaki = datasets.fetch_miyawaki2008()
        
        self.mask = nib.load(self.miyawaki.mask).get_fdata().astype(bool)
        
        if data_split == "train":
            run_indices = range(0, min(24, max_runs))
        else:
            run_indices = range(24, min(32, 24 + max_runs))
            
        all_fmri = []
        all_images = []
        self.img_paths = []
        
        os.makedirs(img_save_dir, exist_ok=True)
        
        trial_counter = 0
        for idx in run_indices:
            func_file = self.miyawaki.func[idx]
            label_file = self.miyawaki.label[idx]
            
            func_data = nib.load(func_file).get_fdata()
            num_trials = func_data.shape[-1]
            
            func_data = np.transpose(func_data, (3, 0, 1, 2))
            masked_voxels = func_data[:, self.mask]
            
            labels = pd.read_csv(label_file, header=None).values
            valid_trials = ~(labels == -1).all(axis=1)
            
            masked_voxels = masked_voxels[valid_trials]
            labels = labels[valid_trials]
            
            all_fmri.append(masked_voxels)
            all_images.append(labels)
            
            # Save the images and record paths
            for l in labels:
                img_10x10 = l.reshape(10, 10).astype(np.float32)
                img_10x10 = (img_10x10 * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_10x10, mode='L').convert('RGB')
                
                # We save the upscaled image so CLIP can process it later
                upscaled = pil_img.resize((224, 224), Image.NEAREST)
                path = os.path.join(img_save_dir, f"{data_split}_{trial_counter:04d}.png")
                upscaled.save(path)
                self.img_paths.append(path)
                trial_counter += 1
            
        self.fmri_data = np.concatenate(all_fmri, axis=0)
        self.image_labels = np.concatenate(all_images, axis=0)
        
        mean = self.fmri_data.mean(axis=0, keepdims=True)
        std = self.fmri_data.std(axis=0, keepdims=True)
        self.fmri_data = (self.fmri_data - mean) / (std + 1e-8)

    def __len__(self):
        return len(self.fmri_data)

    def __getitem__(self, idx):
        fmri_vector = torch.tensor(self.fmri_data[idx], dtype=torch.float32)
        
        # Load the image we saved
        pil_img = Image.open(self.img_paths[idx]).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = transform(pil_img)
        
        return fmri_vector, img_tensor, self.img_paths[idx]

def get_dataloaders(data_dir="mock_data", mode="mock", batch_size=32, num_workers=0):
    """
    Interface expected by phase2_fmri_to_clip.py
    """
    if mode == "miyawaki":
        train_ds = RealFMRIDataset("train", max_runs=24, img_save_dir=os.path.join(data_dir, "miyawaki_images"))
        test_ds = RealFMRIDataset("test", max_runs=8, img_save_dir=os.path.join(data_dir, "miyawaki_images"))
    else:
        # If it asks for mock, we'll just return miyawaki anyway for this biological PoC!
        print(f"Warning: redirecting {mode} to Miyawaki dataset for biological test.")
        train_ds = RealFMRIDataset("train", max_runs=24, img_save_dir=os.path.join(data_dir, "miyawaki_images"))
        test_ds = RealFMRIDataset("test", max_runs=8, img_save_dir=os.path.join(data_dir, "miyawaki_images"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Prep dict (mock scaler, we did Z-score internally)
    prep = {"scaler": None}
    
    return train_loader, test_loader, prep

if __name__ == "__main__":
    tr, ts, _ = get_dataloaders(mode="miyawaki")
    fmri, img, path = next(iter(tr))
    print("Batch fMRI:", fmri.shape, "Batch Img:", img.shape, "Paths:", len(path))
