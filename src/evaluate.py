import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn.manifold import TSNE
import pandas as pd
import warnings

from .facenet_model import get_model
from .traditional import Eigenfaces, Fisherfaces, HOGFaceRecognition
from .data_loader import get_transforms
from .logger_config import setup_logger

warnings.filterwarnings('ignore')
logger = setup_logger("evaluate")


class FaceRecognitionEvaluator:
    def __init__(self, dataset_name='olivetti', test_size=0.3):
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.load_dataset()
        self.results = {}
    
    def load_dataset(self):
        logger.info(f"Loading {self.dataset_name} dataset...")
        
        if self.dataset_name == 'olivetti':
            faces = fetch_olivetti_faces(shuffle=True, random_state=42)
            images = faces.images
            labels = faces.target
            self.target_names = [f"Person_{i:02d}" for i in range(40)]
        else:
            from sklearn.datasets import fetch_lfw_people
            lfw_people = fetch_lfw_people(min_faces_per_person=20, resize=0.5)
            images = lfw_people.images
            labels = lfw_people.target
            self.target_names = lfw_people.target_names
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            images, labels, test_size=self.test_size, random_state=42, stratify=labels
        )
        
        self.num_classes = len(np.unique(labels))
        logger.info(f"Dataset: {len(images)} samples, {self.num_classes} classes")
        logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
    
    def evaluate_traditional_method(self, method_name='eigenfaces'):
        logger.info(f"Evaluating: {method_name.upper()}")
        
        if method_name == 'eigenfaces':
            model = Eigenfaces(n_components=min(150, len(self.X_train) - 1))
        elif method_name == 'fisherfaces':
            model = Fisherfaces()
        elif method_name == 'hog':
            model = HOGFaceRecognition()
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'method': 'traditional'
        }
        
        logger.info(f"Results: Acc={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return accuracy
    
    def evaluate_facenet(self, model_path, loss_name='triplet', model_type='simple', embedding_dim=128):
        logger.info(f"Evaluating FaceNet: {loss_name.upper()}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        need_classifier = loss_name in ['arcface', 'cosface', 'combined']
        num_classes = self.num_classes if need_classifier else None
        
        model = get_model(model_name=model_type, embedding_dim=embedding_dim, num_classes=num_classes, pretrained=False).to(device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Model loaded: {model_path}")
        else:
            logger.warning(f"Model not found: {model_path}, using random init")
        
        model.eval()
        transform = get_transforms(image_size=160, is_training=False)
        
        logger.info("Extracting features...")
        train_embeddings = self._extract_embeddings(self.X_train, model, transform, device)
        test_embeddings = self._extract_embeddings(self.X_test, model, transform, device)
        
        logger.info("Classifying...")
        y_pred = []
        for test_emb in test_embeddings:
            distances = np.linalg.norm(train_embeddings - test_emb, axis=1)
            nearest_idx = np.argmin(distances)
            y_pred.append(self.y_train[nearest_idx])
        
        y_pred = np.array(y_pred)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted', zero_division=0)
        
        method_name = f'facenet_{loss_name}'
        self.results[method_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred,
            'embeddings': test_embeddings,
            'method': 'facenet'
        }
        
        logger.info(f"Results: Acc={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        return accuracy
    
    def _extract_embeddings(self, images, model, transform, device):
        from PIL import Image
        embeddings = []
        
        with torch.no_grad():
            for img in images:
                if img.ndim == 2:
                    img_pil = Image.fromarray((img * 255).astype(np.uint8), mode='L').convert('RGB')
                else:
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
                output = model(img_tensor)
                if isinstance(output, tuple):
                    emb = output[0]
                else:
                    emb = output
                
                embeddings.append(emb.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    def compare_all_methods(self):
        logger.info("Comparing all methods...")
        
        data = []
        for method_name, result in self.results.items():
            data.append({
                'Method': method_name,
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1': f"{result['f1']:.4f}"
            })
        
        df = pd.DataFrame(data)
        logger.info(f"\n{df.to_string(index=False)}")
        return df
    
    def plot_comparison_bar(self, save_path='results/method_comparison.png'):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        methods = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in methods]
        
        method_labels = []
        for m in methods:
            if m == 'eigenfaces':
                method_labels.append('Eigenfaces')
            elif m == 'fisherfaces':
                method_labels.append('Fisherfaces')
            elif m == 'hog':
                method_labels.append('HOG')
            elif m.startswith('facenet_'):
                loss = m.replace('facenet_', '')
                method_labels.append(f'FaceNet\n({loss.upper()})')
            else:
                method_labels.append(m)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#3498db' if 'facenet' in m else '#e74c3c' for m in methods]
        bars = ax.bar(method_labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{acc:.4f}\n({acc*100:.2f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Method Performance Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#e74c3c', alpha=0.8, label='Traditional'),
            Patch(facecolor='#3498db', alpha=0.8, label='Deep Learning')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrix(self, method_name, save_path=None):
        if method_name not in self.results:
            logger.warning(f"Method {method_name} not found")
            return
        
        if save_path is None:
            save_path = f'results/confusion_matrix_{method_name}.png'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        y_pred = self.results[method_name]['y_pred']
        cm = confusion_matrix(self.y_test, y_pred)
        
        if self.num_classes > 20:
            logger.warning(f"Too many classes ({self.num_classes}), skipping confusion matrix")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title(f'Confusion Matrix - {method_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"Confusion matrix saved: {save_path}")
        plt.close()
    
    def plot_tsne(self, method_name, save_path=None, n_samples=200):
        if method_name not in self.results:
            logger.warning(f"Method {method_name} not found")
            return
        
        if 'embeddings' not in self.results[method_name]:
            logger.warning(f"Method {method_name} has no embeddings")
            return
        
        if save_path is None:
            save_path = f'results/tsne_{method_name}.png'
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        embeddings = self.results[method_name]['embeddings']
        
        if len(embeddings) > n_samples:
            indices = np.random.choice(len(embeddings), n_samples, replace=False)
            embeddings_sampled = embeddings[indices]
            labels_sampled = self.y_test[indices]
        else:
            embeddings_sampled = embeddings
            labels_sampled = self.y_test
        
        logger.info(f"Running t-SNE on {len(embeddings_sampled)} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_sampled)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels_sampled, cmap='tab20', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Class ID')
        plt.title(f't-SNE Visualization - {method_name}', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dim 1', fontsize=12)
        plt.ylabel('t-SNE Dim 2', fontsize=12)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        logger.info(f"t-SNE plot saved: {save_path}")
        plt.close()


def main_evaluation():
    logger.info("=== Face Recognition Evaluation ===")
    
    evaluator = FaceRecognitionEvaluator(dataset_name='olivetti', test_size=0.3)
    
    evaluator.evaluate_traditional_method('eigenfaces')
    evaluator.evaluate_traditional_method('fisherfaces')
    evaluator.evaluate_traditional_method('hog')
    
    facenet_models = [
        ('models/simple_triplet_best.pth', 'triplet', 'simple', 128),
        ('models/simple_arcface_best.pth', 'arcface', 'simple', 128),
        ('models/simple_center_best.pth', 'center', 'simple', 128),
    ]
    
    for model_path, loss_name, model_type, emb_dim in facenet_models:
        if os.path.exists(model_path):
            evaluator.evaluate_facenet(model_path, loss_name, model_type, emb_dim)
        else:
            logger.warning(f"Model not found: {model_path}")
    
    df = evaluator.compare_all_methods()
    
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/comparison_table.csv', index=False)
    logger.info("Comparison table saved: results/comparison_table.csv")
    
    evaluator.plot_comparison_bar()
    
    for method_name in list(evaluator.results.keys())[:3]:
        evaluator.plot_confusion_matrix(method_name)
    
    for method_name in evaluator.results.keys():
        if method_name.startswith('facenet_'):
            evaluator.plot_tsne(method_name)
    
    logger.info("Evaluation complete!")

