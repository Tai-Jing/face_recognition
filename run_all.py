#!/usr/bin/env python
"""一键运行完整实验流程"""
import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from src.logger_config import setup_logger

logger = setup_logger("run_all")


def run_command(cmd):
    logger.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        return False
    return True


def main():
    logger.info("=" * 60)
    logger.info("Face Recognition Experiment - Complete Pipeline")
    logger.info("=" * 60)
    
    logger.info("\n[1/7] Training Eigenfaces...")
    run_command("python main.py train-traditional --algorithm eigenfaces")
    
    logger.info("\n[2/7] Training Fisherfaces...")
    run_command("python main.py train-traditional --algorithm fisherfaces")
    
    logger.info("\n[3/7] Training HOG...")
    run_command("python main.py train-traditional --algorithm hog")
    
    logger.info("\n[4/7] Training FaceNet with Triplet Loss...")
    run_command("python main.py train-deep --loss triplet --epochs 50")
    
    logger.info("\n[5/7] Training FaceNet with Center Loss...")
    run_command("python main.py train-deep --loss center --epochs 50")
    
    logger.info("\n[6/7] Training FaceNet with ArcFace Loss...")
    run_command("python main.py train-deep --loss arcface --epochs 50")
    
    logger.info("\n[7/7] Evaluating all methods...")
    run_command("python main.py evaluate")
    
    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info("Check results/ directory for visualizations and metrics.")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

