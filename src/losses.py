import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .logger_config import setup_logger

logger = setup_logger("losses")


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5, distance='euclidean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(self, anchor, positive, negative):
        if self.distance == 'euclidean':
            dist_ap = F.pairwise_distance(anchor, positive, p=2)
            dist_an = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance == 'cosine':
            dist_ap = 1 - F.cosine_similarity(anchor, positive)
            dist_an = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance: {self.distance}")
        
        losses = F.relu(dist_ap - dist_an + self.margin)
        return losses.mean()


class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device='cuda'):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))
    
    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels.long())
        loss = F.mse_loss(features, centers_batch)
        return loss


class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.50, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(features, weight)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        loss = F.cross_entropy(output, labels)
        return loss


class CosineFaceLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, s=30.0, m=0.35):
        super(CosineFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.s = s
        self.m = m
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(features, weight)
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = cosine - one_hot * self.m
        output *= self.s
        
        loss = F.cross_entropy(output, labels)
        return loss


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, center_weight=0.01, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(num_classes, feat_dim, device)
        self.center_weight = center_weight
    
    def forward(self, logits, features, labels):
        loss_softmax = self.softmax_loss(logits, labels)
        loss_center = self.center_loss(features, labels)
        total_loss = loss_softmax + self.center_weight * loss_center
        return total_loss, loss_softmax, loss_center


def get_loss_function(loss_name, num_classes=None, feat_dim=512, device='cuda', **kwargs):
    loss_name = loss_name.lower()
    
    if loss_name == 'triplet':
        margin = kwargs.get('margin', 0.5)
        distance = kwargs.get('distance', 'euclidean')
        return TripletLoss(margin=margin, distance=distance)
    elif loss_name == 'center':
        if num_classes is None:
            raise ValueError("Center Loss requires num_classes")
        return CenterLoss(num_classes, feat_dim, device)
    elif loss_name == 'arcface':
        if num_classes is None:
            raise ValueError("ArcFace Loss requires num_classes")
        s = kwargs.get('s', 30.0)
        m = kwargs.get('m', 0.50)
        return ArcFaceLoss(num_classes, feat_dim, s=s, m=m)
    elif loss_name == 'cosface':
        if num_classes is None:
            raise ValueError("CosineFace Loss requires num_classes")
        s = kwargs.get('s', 30.0)
        m = kwargs.get('m', 0.35)
        return CosineFaceLoss(num_classes, feat_dim, s=s, m=m)
    elif loss_name == 'combined':
        if num_classes is None:
            raise ValueError("Combined Loss requires num_classes")
        center_weight = kwargs.get('center_weight', 0.01)
        return CombinedLoss(num_classes, feat_dim, center_weight, device)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

