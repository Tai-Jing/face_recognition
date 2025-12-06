import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.datasets import fetch_lfw_people, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from .logger_config import setup_logger

logger = setup_logger("data_loader")


class FaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if image.ndim == 2:
            image = Image.fromarray((image * 255).astype(np.uint8), mode='L').convert('RGB')
        elif image.ndim == 3 and image.shape[2] == 3:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class TripletFaceDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
        
        self.valid_labels = [label for label, indices in self.label_to_indices.items() if len(indices) >= 2]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        anchor_label = self.labels[idx]
        anchor_img = self._load_image(idx)
        
        positive_indices = [i for i in self.label_to_indices[anchor_label] if i != idx]
        positive_idx = idx if len(positive_indices) == 0 else np.random.choice(positive_indices)
        positive_img = self._load_image(positive_idx)
        
        negative_label = anchor_label
        while negative_label == anchor_label:
            negative_label = np.random.choice(self.valid_labels)
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative_img = self._load_image(negative_idx)
        
        return (anchor_img, positive_img, negative_img), anchor_label
    
    def _load_image(self, idx):
        image = self.images[idx]
        
        if image.ndim == 2:
            image = Image.fromarray((image * 255).astype(np.uint8), mode='L').convert('RGB')
        elif image.ndim == 3 and image.shape[2] == 3:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image


def download_lfw_dataset(data_dir='./data/lfw', min_faces_per_person=20):
    logger.info("Loading LFW dataset...")
    try:
        lfw_people = fetch_lfw_people(data_home=data_dir, min_faces_per_person=min_faces_per_person, resize=0.5, color=False)
        images = lfw_people.images
        labels = lfw_people.target
        target_names = lfw_people.target_names
        logger.info(f"LFW dataset loaded: {len(images)} samples, {len(target_names)} persons")
        return images, labels, target_names
    except Exception as e:
        logger.warning(f"LFW download failed: {e}, using Olivetti Faces instead")
        return download_olivetti_dataset(data_dir)


def download_olivetti_dataset(data_dir='./data/olivetti'):
    logger.info("Loading Olivetti Faces dataset...")
    os.makedirs(data_dir, exist_ok=True)
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    images = faces.images
    labels = faces.target
    target_names = [f"Person_{i:02d}" for i in range(40)]
    logger.info(f"Olivetti dataset loaded: {len(images)} samples, {len(target_names)} persons")
    return images, labels, target_names


def get_transforms(image_size=160, is_training=True):
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    return transform


def prepare_dataloaders(dataset_name='olivetti', batch_size=32, test_size=0.3, image_size=160, use_triplet=False):
    if dataset_name == 'lfw':
        images, labels, target_names = download_lfw_dataset()
    else:
        images, labels, target_names = download_olivetti_dataset()
    
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42, stratify=labels)
    logger.info(f"Dataset split: train={len(X_train)}, test={len(X_test)}")
    
    train_transform = get_transforms(image_size, is_training=True)
    test_transform = get_transforms(image_size, is_training=False)
    
    if use_triplet:
        train_dataset = TripletFaceDataset(X_train, y_train, transform=train_transform)
        test_dataset = TripletFaceDataset(X_test, y_test, transform=test_transform)
    else:
        train_dataset = FaceDataset(X_train, y_train, transform=train_transform)
        test_dataset = FaceDataset(X_test, y_test, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    num_classes = len(target_names)
    
    return train_loader, test_loader, num_classes, target_names

