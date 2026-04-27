# Spike-Attention
> Hybrid spiking neural network with self-attention for skin lesion severity classification

## Overview
This repository implements a Spiking Neural Network (SNN)-based image classification framework with a custom attention mechanism.

The model integrates:
- ResNet backbone
- Spike-based attention module (ATT)
- Deformable convolution
- Temporal spike accumulation (multi-step simulation)

The goal is to improve feature representation and robustness by combining:
- spatial attention
- global attention
- spike driven computation

---

## Dataset Information
Datasets are not included and must be prepared manually.
```
data/
├── train/
 │ ├── class0/
 │ ├── class1/
 │ └── ...
├── test/
 │ ├── class0/
 │ ├── class1/
 │ └── ...
```
### Details
- number of classes : 5
- Input size: 100×100
- Format: RGB images

(This implementation is designed for a custom 5-class classification task)

## Code Information
### Main Components
1. Model Architecture
   - ResNetSNN
     - Modified ResNet with spike-based forward pass
     - Temporal simulation over multiple steps
   - ATT
     - Global attention (channel-wise)
     - Multi-scale spatial attention
     - Deformable convolution
     - Spike neuron (LIF)
   - DeformableConv2d
      - Adaptive spatial feature extraction
2. Training Pipeline
   - Optimizer
     - AdamW
   - Loss
     - Label Smoothing Cross Entropy
   - Scheduler
     - Warmup + Cosine Annealing
   
## Usage Instructions
1. Prepare dataset
   - Place dataset in:
      - ./data/train/
      - ./data/test/
2. Install dependencies
```
pip install -r requirements.txt
```
3. Train model
```
python att_sgsls.py
```
5. Key parameters
 - Epochs: 50
 - Batch size: 8
 - steps: 30
 - beta: 0.5  
5. Outputs
   - Evaluation metrics and visualizations are automatically generated after training
   - Results are saved in:
```
./checkpoint/att-sgsls-30/
```
   - Including:
     - Model checkpoint (.pth)
     - Learning curve
     - Confusion matrix
     - ROC curve
     - Grad-CAM visualizations

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- timm
- snntorch
  
**GPU recommended (CUDA support)**

## Methodology
1. Input image → ResNet backbone
2. Feature maps pass through ATT modules
3. Spike neurons (LIF) simulate temporal dynamics
4. Features accumulated over multiple time steps
5. Final classification via fully connected layer

### Key Features
- Spike-based computation
  - Uses LIF neurons
- Attention mechanism
  - Combines global + spatial attention
- Temporal modeling
  - Multi-spike accumulation (e.g. 30)
- Deformable convolution
  - Adaptive receptive fields

## License
**MIT license**

#### Note
This work builds upon established architectures and presents a novel integration of spike-based attention mechanisms with task-specific fine-tuning.
