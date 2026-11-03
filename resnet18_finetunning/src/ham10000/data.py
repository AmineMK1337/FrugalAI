from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

class Ham10000Dataset(Dataset):
    """PyTorch Dataset for the HAM10000 skin lesion images.

    Reads the metadata CSV, maps each row to its image file (handling the
    two image folders the dataset ships in), and returns (image_tensor,
    label_int) pairs. String diagnoses are mapped to integer labels via a
    deterministic, sorted class_to_idx mapping built at construction.
    """
    
    def __init__(self, csv_path, image_dir, transform=None):
        """
        Args:
            csv_path: Path to HAM10000_metadata.csv.
            image_dir: Directory containing the two HAM10000 image folders.
            transform: Optional torchvision transform applied to each image.
        """
        
        self.data = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform

        # Normalize column names if using ISIC metadata format
        if 'image_id' not in self.data.columns and 'isic_id' in self.data.columns:
            self.data['image_id'] = self.data['isic_id']

        if 'dx' not in self.data.columns and 'diagnosis_3' in self.data.columns:
            def map_dx(r):
                d3 = str(r.get('diagnosis_3', ''))
                d2 = str(r.get('diagnosis_2', ''))
                if 'Nevus' in d3:
                    return 'nv'
                elif 'Pigmented benign keratosis' in d3:
                    return 'bkl'
                elif 'Melanoma' in d3:
                    return 'mel'
                elif 'Basal cell' in d3:
                    return 'bcc'
                elif 'Squamous cell' in d3 or 'actinic' in d3:
                    return 'akiec'
                elif 'Vascular' in d2:
                    return 'vasc'
                elif 'Dermatofibroma' in d3:
                    return 'df'
                return 'unknown'
            self.data['dx'] = self.data.apply(map_dx, axis=1)

        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.data['dx'].unique()))}
        self.image_paths = {}
        # Support both original layout (part_1/part_2), flat ISIC-images layout, and class-folder layout
        search_dirs = ['HAM10000_images_part_1', 'HAM10000_images_part_2', 'ISIC-images']
        search_dirs += sorted(self.data['dx'].str.upper().unique().tolist())
        for part in search_dirs:
            folder = self.image_dir / part
            if folder.exists():
                for p in folder.glob('*.jpg'):
                    self.image_paths[p.stem] = p

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Return (image_tensor, label_int) for the row at position idx."""
        
        row = self.data.iloc[idx]
        path = self.image_paths.get(row['image_id'])
        if path is None:
            raise FileNotFoundError(f"Image not found for id: {row['image_id']}")
        img = Image.open(path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        label = self.class_to_idx[row['dx']]
        
        return img, int(label)


def eval_transform(config):
    """Deterministic resize + normalize pipeline used for val and test."""
    return transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])


def lesion_split(data, config):
    """Return (train_idx, val_idx, test_idx) row indices of the fixed split.

    The partitions come from config.splits_csv, written by make_splits.py: 70/15/15
    of the lesions, stratified by dx. Every image follows its lesion.
    """
    partition = pd.read_csv(config.splits_csv).set_index("image_id")["partition"]
    row_partition = data["image_id"].map(partition)
    if len(partition) != len(data) or row_partition.isna().any():
        raise ValueError(f"{config.splits_csv} does not cover the metadata rows exactly")
    return tuple(data.index[row_partition == p].tolist() for p in ("train", "validation", "test"))


def build_test_loader(config, batch_size=None, num_workers=0):
    """Build the test DataLoader: no augmentation, no shuffling.

    Uses the same lesion_split as build_dataloaders, so it is exactly the
    15% of lesions that training never saw.

    Returns:
        (test_loader, test_df, class_to_idx) where test_df holds the test
        rows of the metadata in loader order.
    """
    test_ds = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=eval_transform(config))
    _, _, test_idx = lesion_split(test_ds.data, config)
    test_loader = DataLoader(Subset(test_ds, test_idx), batch_size=batch_size or config.batch_size,
                             shuffle=False, num_workers=num_workers)
    return test_loader, test_ds.data.loc[test_idx].reset_index(drop=True), test_ds.class_to_idx


def build_dataloaders(config):
    """Build train and val DataLoaders from the fixed 70/15/15 stratified split.

    The split is at lesion level, so multiple images of the same lesion never
    appear in more than one partition, which would leak information. The
    remaining 15% test partition is not loaded here. Training data is augmented when config.use_augmentation is set (config.augmentation picks
    "standard" or "strong"); validation data always uses a deterministic
    resize + normalize pipeline.

    Returns:
        (train_loader, val_loader, class_to_idx)
    """

    if config.augmentation == "strong":
        # Dermoscopy has no canonical orientation, so any rotation and both
        # flips are label-preserving; colour jitter stays mild because colour
        # is diagnostic.
        augment = [
            transforms.RandomRotation(degrees=180),
            transforms.RandomResizedCrop(config.image_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
        ]
    else:
        augment = [
            transforms.RandomResizedCrop(config.image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ]
    train_transform = transforms.Compose([
        *augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    val_transform = eval_transform(config)
    chosen_train_transform = train_transform if config.use_augmentation else val_transform
    train_ds = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=chosen_train_transform)
    val_ds   = Ham10000Dataset(config.metadata_csv, config.data_dir, transform=val_transform)

    train_idx, val_idx, _test_idx = lesion_split(train_ds.data, config)

    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=torch.cuda.is_available(),
                         persistent_workers=config.num_workers > 0)
    train_loader = DataLoader(Subset(train_ds, train_idx), shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(Subset(val_ds, val_idx), shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_ds.class_to_idx


def compute_class_weights(train_loader, num_classes, device):
    """Compute inverse-frequency class weights from the training split.

    Uses sklearn's 'balanced' scheme so rarer classes receive larger weights,
    counteracting the dataset's heavy imbalance. Weights are computed from the
    training labels only (never val) to avoid leakage, and returned as a float
    tensor on the given device for use in nn.CrossEntropyLoss(weight=...).

    Returns:
        A 1-D tensor of length num_classes, ordered by class index.
    """
    
    subset = train_loader.dataset
    full_df = subset.dataset.data
    class_to_idx = subset.dataset.class_to_idx
    train_labels = full_df.iloc[list(subset.indices)]['dx'].map(class_to_idx).to_numpy()
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=train_labels
    )
    return torch.tensor(weights, dtype=torch.float32).to(device)