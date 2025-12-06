import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .logger_config import setup_logger

logger = setup_logger("facenet_model")


class InceptionResnetV1(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=None, pretrained=False):
        super(InceptionResnetV1, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)
        
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        feature_dim = 2048
        
        self.embedding = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        if self.classifier is not None:
            logits = self.classifier(embeddings_normalized)
            return embeddings_normalized, logits
        else:
            return embeddings_normalized
    
    def get_embedding(self, x):
        with torch.no_grad():
            embeddings = self.forward(x)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
        return embeddings


class SimpleCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=None):
        super(SimpleCNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_dim, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding(features)
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        if self.classifier is not None:
            logits = self.classifier(embeddings_normalized)
            return embeddings_normalized, logits
        else:
            return embeddings_normalized
    
    def get_embedding(self, x):
        with torch.no_grad():
            embeddings = self.forward(x)
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]
        return embeddings


def get_model(model_name='simple', embedding_dim=512, num_classes=None, pretrained=False):
    if model_name == 'simple':
        return SimpleCNN(embedding_dim=embedding_dim, num_classes=num_classes)
    elif model_name == 'resnet':
        return InceptionResnetV1(embedding_dim=embedding_dim, num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")

