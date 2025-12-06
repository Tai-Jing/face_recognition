import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.train import train_facenet
from src.evaluate import FaceRecognitionEvaluator, main_evaluation
from src.traditional import Eigenfaces, Fisherfaces, HOGFaceRecognition, evaluate_traditional_method, visualize_eigenfaces
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from src.logger_config import setup_logger

logger = setup_logger("main")


def train_traditional(args):
    logger.info(f"Training {args.algorithm.upper()}...")
    
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        faces.images, faces.target, test_size=0.3, random_state=42, stratify=faces.target
    )
    
    if args.algorithm == 'eigenfaces':
        model = Eigenfaces(n_components=min(150, len(X_train) - 1))
    elif args.algorithm == 'fisherfaces':
        model = Fisherfaces()
    elif args.algorithm == 'hog':
        model = HOGFaceRecognition()
    else:
        logger.error(f"Unknown algorithm: {args.algorithm}")
        return
    
    accuracy, _ = evaluate_traditional_method(model, X_train, y_train, X_test, y_test, args.algorithm.capitalize())
    
    if args.algorithm == 'eigenfaces':
        eigenfaces = model.get_eigenfaces(n_faces=10)
        visualize_eigenfaces(eigenfaces)
    
    logger.info(f"Accuracy: {accuracy:.4f}")


def train_deep(args):
    logger.info(f"Training FaceNet with {args.loss.upper()} Loss...")
    
    loss_params = {}
    if args.loss == 'triplet':
        loss_params = {'margin': 0.5}
    elif args.loss == 'arcface':
        loss_params = {'s': 30.0, 'm': 0.50}
    elif args.loss == 'center':
        loss_params = {}
    
    trainer, history = train_facenet(
        loss=args.loss,
        dataset='olivetti',
        epochs=args.epochs,
        batch_size=32,
        lr=0.001,
        model='simple',
        embedding_dim=128,
        pretrained=False,
        loss_params=loss_params
    )
    
    logger.info(f"Training complete! Best accuracy: {max(history['val_accuracy']):.4f}")


def evaluate(args):
    logger.info("Starting evaluation...")
    main_evaluation()
    logger.info("Evaluation complete! Check results/ directory.")


def main():
    parser = argparse.ArgumentParser(description='Face Recognition: Traditional vs Deep Learning')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train traditional methods
    train_trad = subparsers.add_parser('train-traditional', help='Train traditional methods (Eigenfaces/Fisherfaces/HOG)')
    train_trad.add_argument('--algorithm', type=str, required=True, choices=['eigenfaces', 'fisherfaces', 'hog'])
    
    # Train deep learning methods
    train_dl = subparsers.add_parser('train-deep', help='Train FaceNet with different losses')
    train_dl.add_argument('--loss', type=str, required=True, choices=['triplet', 'center', 'arcface'])
    train_dl.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    
    # Evaluate all methods
    subparsers.add_parser('evaluate', help='Evaluate and compare all methods')
    
    args = parser.parse_args()
    
    if args.command == 'train-traditional':
        train_traditional(args)
    elif args.command == 'train-deep':
        train_deep(args)
    elif args.command == 'evaluate':
        evaluate(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

