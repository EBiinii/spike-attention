# Spike-Attention
> Hybrid spiking neural network with self-attention for skin lesion severity classification

## Abstract
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

## Framework

### Architecture Overview

The proposed **Spike-Attention** framework combines a convolutional backbone, spike-based attention mechanisms, and temporal neural dynamics for skin lesion severity classification.

```text
Input Image
      │
      ▼
 ResNet Backbone
      │
      ▼
 Spike-Attention Module
 ├─ Global Channel Attention
 ├─ Multi-Scale Spatial Attention
 ├─ Deformable Convolution
 └─ LIF Spiking Neurons
      │
      ▼
 Temporal Spike Accumulation
      │
      ▼
 Fully Connected Layer
      │
      ▼
 Severity Classification
```

The network first extracts visual features using a modified ResNet backbone. The extracted feature maps are then refined through the proposed **Spike-Attention (ATT)** module, which integrates:

- Global channel attention
- Multi-scale spatial attention
- Deformable convolution
- Leaky Integrate-and-Fire (LIF) neurons

The attention-enhanced features are propagated through multiple simulation steps, allowing temporal information to accumulate over time. The aggregated spike representations are then passed to a fully connected layer for final classification.

### Key Components

| Component | Description |
|------------|-------------|
| ResNet Backbone | Extracts hierarchical visual features |
| Global Attention | Captures channel-wise contextual information |
| Spatial Attention | Emphasizes informative lesion regions |
| Deformable Convolution | Adapts receptive fields to irregular lesion structures |
| LIF Neurons | Introduce spike-based temporal dynamics |
| Temporal Accumulation | Aggregates spike responses across multiple time steps |

### Processing Pipeline

1. Input image preprocessing
2. Feature extraction using ResNet backbone
3. Feature refinement through Spike-Attention modules
4. Spike generation via LIF neurons
5. Temporal accumulation over multiple simulation steps
6. Final classification through a fully connected layer


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

**Note:** The dataset collected under DRB approval and does not contain directly identifiable personal information. Due to institutional data governance policies, the dataset is available upon reasonable request for approved academic research. To support reproducibility, we release the full codebase, preprocessing pipeline, and data format specifications, and demonstrate that the proposed method can be applied to other publicly available datasets.

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

---

#### Note
This work builds upon established architectures and presents a novel integration of spike-based attention mechanisms with task-specific fine-tuning.
