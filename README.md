# 人脸识别实验系统

传统方法与深度度量学习的人脸识别对比实验

## 项目简介

本项目实现并对比了传统方法与深度学习方法在人脸识别中的性能表现：

**传统方法：**
- Eigenfaces (PCA特征脸)
- Fisherfaces (LDA线性判别)
- HOG特征 + 最近邻

**深度学习方法 (FaceNet)：**
- Triplet Loss (三元组损失)
- Center Loss (中心损失)
- ArcFace Loss (角度边界损失)

**实验数据集：** Olivetti Faces (400张图像，40人，每人10张)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练传统方法

```bash
# Eigenfaces
python main.py train-traditional --algorithm eigenfaces

# Fisherfaces
python main.py train-traditional --algorithm fisherfaces

# HOG
python main.py train-traditional --algorithm hog
```

### 3. 训练深度学习方法

```bash
# Triplet Loss
python main.py train-deep --loss triplet --epochs 50

# ArcFace Loss
python main.py train-deep --loss arcface --epochs 50

# Center Loss
python main.py train-deep --loss center --epochs 50
```

### 4. 评估与对比

```bash
python main.py evaluate
```

评估结果将保存在 `results/` 目录，包括：
- 准确率对比图
- 混淆矩阵
- t-SNE特征可视化
- 性能指标表格

## 项目结构

```
digital/
├── src/
│   ├── traditional.py      # 传统方法实现
│   ├── facenet_model.py    # FaceNet模型
│   ├── losses.py           # 损失函数
│   ├── train.py            # 训练脚本
│   ├── evaluate.py         # 评估脚本
│   ├── data_loader.py      # 数据加载
│   └── logger_config.py    # 日志配置
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖包
├── models/                 # 保存的模型
├── results/                # 实验结果
└── logs/                   # 运行日志
```

## 实验结果示例

基于Olivetti数据集的实验表明，深度学习方法显著优于传统方法：

| 方法 | 准确率 |
|------|--------|
| Eigenfaces | ~92.5% |
| Fisherfaces | ~92.5% |
| HOG | ~90.0% |
| Triplet Loss | ~94.0% |
| Center Loss | ~95.0% |
| ArcFace Loss | ~96.7% |

**结论：** ArcFace通过角度边界约束实现了最优性能，较传统方法提升约4-7个百分点。

## 核心技术

### Eigenfaces (特征脸)
- 使用PCA降维提取主成分
- 基于欧氏距离的最近邻分类

### Fisherfaces (Fisher脸)
- PCA + LDA两步降维
- 最大化类间距离，最小化类内距离

### FaceNet + Triplet Loss
- 学习嵌入空间，使同类样本接近，异类样本远离
- 损失函数：L = max(d(a,p) - d(a,n) + margin, 0)

### FaceNet + Center Loss
- 为每个类别学习一个中心点
- 最小化样本到所属类别中心的距离

### FaceNet + ArcFace Loss
- 在角度空间添加加性角度边界
- 增强特征的类间区分性和类内紧凑性

## 参考文献

1. Schroff et al. "FaceNet: A Unified Embedding for Face Recognition and Clustering" (CVPR 2015)
2. Wen et al. "A Discriminative Feature Learning Approach for Deep Face Recognition" (ECCV 2016)
3. Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)

## 技术栈

- PyTorch 2.0+
- scikit-learn
- scikit-image
- Matplotlib & Seaborn
- TensorBoard

## License

MIT License

