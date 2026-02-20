import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os


def get_transforms(mode='train'):
    """
    Data augmentation pipeline.
    Train: aggressive augmentation to prevent overfitting
    Val: minimal preprocessing only
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),  # Randomly erase patches
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


class MultilabelDataset(Dataset):
    """
    Dataset for multilabel classification with NA handling via masking.
    Returns: (image, label, mask) where mask=0 for NA positions.
    """
    
    def __init__(self, labels_file, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.samples = []
        self._load_labels(labels_file)
    
    def _load_labels(self, labels_file):
        skipped = 0
        with open(labels_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                filename = parts[0]
                img_path = os.path.join(self.images_dir, filename)
                
                if not os.path.exists(img_path):
                    skipped += 1
                    continue
                
                label, mask = [], []
                for val in parts[1:]:
                    if val.upper() == 'NA':
                        label.append(0.0)
                        mask.append(0.0)
                    else:
                        label.append(float(val))
                        mask.append(1.0)
                
                self.samples.append((
                    filename,
                    torch.tensor(label, dtype=torch.float32),
                    torch.tensor(mask, dtype=torch.float32)
                ))
        
        if skipped > 0:
            print(f"Skipped {skipped} missing images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, label, mask = self.samples[idx]
        img_path = os.path.join(self.images_dir, filename)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, mask


def get_dataloaders(labels_file, images_dir, val_split=0.2, batch_size=32, num_workers=0, seed=42):
    """
    Creates train/val dataloaders with proper transforms.
    """
    full_dataset = MultilabelDataset(labels_file, images_dir, get_transforms('train'))
    
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Apply validation transforms to val set
    val_set.dataset = MultilabelDataset(labels_file, images_dir, get_transforms('val'))
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    print(f"Train: {train_size} | Val: {val_size}")
    return train_loader, val_loader