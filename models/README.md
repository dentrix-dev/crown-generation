# Models

This repository implements various deep learning models for processing 3D point clouds, focusing on tasks like reconstruction and completion. Below is a summary of the models implemented and planned for this repository.

---

### 1. FoldingNet
FoldingNet is an autoencoder designed for unsupervised feature learning on 3D point clouds. It reconstructs the input point cloud by "folding" a 2D grid onto a 3D surface.

- **Paper**: [FoldingNet: Point Cloud Auto-encoder via Deep Grid Deformation](https://arxiv.org/abs/1712.07262)
- **Current Implementation**: FoldingNet is implemented as a standalone Point Cloud Generator.
- **Future Enhancements**:
  - Integration with other attention mechanisms and DGCNN for crown generation

![FoldingNet](/images/Graphs/FoldingNet.png)

---

### 2. PoinTr

UPWORKING


## Usage  
This repository provides a unified interface for all implemented models. You can easily switch between models and tasks using the factory pattern.

Import the desired model in your script:  
```python  
from factories.model_factory import get_model

# Example: Load DynamicGraphCNN for segmentation task with 33 output classes
model = get_model("DynamicGraphCNN", "segmentation", 33)
