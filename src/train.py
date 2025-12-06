import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from .facenet_model import get_model
from .losses import get_loss_function
from .data_loader import prepare_dataloaders
from .logger_config import setup_logger

logger = setup_logger("train")


class FaceNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Device: {self.device}")
        
        self.prepare_data()
        self.create_model()
        self.create_loss_function()
        self.create_optimizer()
        
        self.writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/facenet'))
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def prepare_data(self):
        logger.info("Preparing dataset...")
        use_triplet = (self.config['loss'] == 'triplet')
        
        self.train_loader, self.val_loader, self.num_classes, self.target_names = prepare_dataloaders(
            dataset_name=self.config.get('dataset', 'olivetti'),
            batch_size=self.config.get('batch_size', 32),
            test_size=self.config.get('test_size', 0.3),
            image_size=self.config.get('image_size', 160),
            use_triplet=use_triplet
        )
        logger.info(f"Num classes: {self.num_classes}")
    
    def create_model(self):
        logger.info("Creating model...")
        loss_name = self.config['loss']
        need_classifier = loss_name in ['arcface', 'cosface', 'combined']
        num_classes = self.num_classes if need_classifier else None
        
        self.model = get_model(
            model_name=self.config.get('model', 'simple'),
            embedding_dim=self.config.get('embedding_dim', 128),
            num_classes=num_classes,
            pretrained=self.config.get('pretrained', False)
        ).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
    
    def create_loss_function(self):
        logger.info(f"Creating loss function: {self.config['loss']}")
        self.criterion = get_loss_function(
            loss_name=self.config['loss'],
            num_classes=self.num_classes,
            feat_dim=self.config.get('embedding_dim', 128),
            device=self.device,
            **self.config.get('loss_params', {})
        )
        
        if self.config['loss'] in ['arcface', 'cosface']:
            self.criterion = self.criterion.to(self.device)
    
    def create_optimizer(self):
        lr = self.config.get('lr', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.get('lr_step', 10), gamma=self.config.get('lr_gamma', 0.1))
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
        
        for batch_idx, batch_data in enumerate(pbar):
            if self.config['loss'] == 'triplet':
                (anchor, positive, negative), labels = batch_data
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)
                loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                
            elif self.config['loss'] == 'combined':
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                embeddings, logits = self.model(images)
                loss, loss_softmax, loss_center = self.criterion(logits, embeddings, labels)
                
            elif self.config['loss'] in ['arcface', 'cosface']:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                embeddings, _ = self.model(images)
                loss = self.criterion(embeddings, labels)
                
            else:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                embeddings = self.model(images)
                if isinstance(embeddings, tuple):
                    embeddings = embeddings[0]
                loss = self.criterion(embeddings, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                if self.config['loss'] == 'triplet':
                    (anchor, positive, negative), labels = batch_data
                    anchor = anchor.to(self.device)
                    positive = positive.to(self.device)
                    negative = negative.to(self.device)
                    
                    anchor_emb = self.model(anchor)
                    positive_emb = self.model(positive)
                    negative_emb = self.model(negative)
                    loss = self.criterion(anchor_emb, positive_emb, negative_emb)
                    
                    all_embeddings.append(anchor_emb.cpu().numpy())
                    all_labels.append(labels.numpy())
                    
                else:
                    images, labels = batch_data
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    output = self.model(images)
                    
                    if isinstance(output, tuple):
                        embeddings, logits = output
                        if self.config['loss'] == 'combined':
                            loss, _, _ = self.criterion(logits, embeddings, labels)
                        elif self.config['loss'] in ['arcface', 'cosface']:
                            loss = self.criterion(embeddings, labels)
                        else:
                            loss = self.criterion(embeddings, labels)
                    else:
                        embeddings = output
                        loss = self.criterion(embeddings, labels)
                    
                    all_embeddings.append(embeddings.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.concatenate(all_labels)
        accuracy = self.compute_accuracy(all_embeddings, all_labels)
        
        return avg_loss, accuracy
    
    def compute_accuracy(self, embeddings, labels, k=1):
        n_samples = len(embeddings)
        correct = 0
        
        for i in range(n_samples):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[i] = False
            
            X_train = embeddings[train_mask]
            y_train = labels[train_mask]
            X_test = embeddings[i:i+1]
            y_test = labels[i]
            
            distances = np.linalg.norm(X_train - X_test, axis=1)
            pred = y_train[np.argmin(distances)]
            
            if pred == y_test:
                correct += 1
        
        accuracy = correct / n_samples
        return accuracy
    
    def train(self):
        logger.info(f"Start training - Loss: {self.config['loss']}")
        best_accuracy = 0.0
        best_epoch = 0
        
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_loss, val_accuracy = self.validate(epoch)
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/val', val_accuracy, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.config['epochs']}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}, lr={current_lr:.6f}")
            
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                self.save_checkpoint('best')
                logger.info("Best model saved")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}')
        
        logger.info(f"Training completed! Best val accuracy: {best_accuracy:.4f} (Epoch {best_epoch+1})")
        self.save_checkpoint('final')
        self.plot_training_curves()
        
        return self.history
    
    def save_checkpoint(self, name='checkpoint'):
        save_dir = self.config.get('save_dir', 'models')
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        loss_name = self.config['loss']
        model_name = self.config.get('model', 'simple')
        filename = f'{model_name}_{loss_name}_{name}.pth'
        filepath = os.path.join(save_dir, filename)
        torch.save(checkpoint, filepath)
    
    def plot_training_curves(self):
        save_dir = self.config.get('save_dir', 'results')
        os.makedirs(save_dir, exist_ok=True)
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title(f'Training Curves - {self.config["loss"]} Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, self.history['val_accuracy'], 'g-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        loss_name = self.config['loss']
        filename = f'training_curves_{loss_name}.png'
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches='tight')
        logger.info(f"Training curves saved: {filepath}")
        plt.close()


def train_facenet(loss='triplet', dataset='olivetti', epochs=50, batch_size=32, lr=0.001, model='simple', embedding_dim=128, **kwargs):
    config = {
        'loss': loss,
        'dataset': dataset,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'model': model,
        'embedding_dim': embedding_dim,
        'save_dir': 'models',
        'log_dir': f'runs/facenet_{loss}',
        **kwargs
    }
    
    trainer = FaceNetTrainer(config)
    history = trainer.train()
    
    return trainer, history

