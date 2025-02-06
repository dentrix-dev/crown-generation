# CrownGeneration
AI Pipeline for Crown Generation Module

## **📌 Our Architecture**
![OurArchitecture](/images/Pipeline.png)

## **📌 Abstract Representation of our Architecture**
Our Architecture is a **transformer-based** encoder-decoder model for **point cloud completion**.  

1. **Input:**  
   - **Incomplete point cloud** → \( X \in \mathbb{R}^{B \times N \times 3} \)  
   - Where \( B \) = batch size, \( N \) = number of points, and **each point has 3D coordinates (x, y, z)**.  

### **1️⃣ Feature Extraction (DGCNN)**
- **Extracts local features using a Dynamic Graph CNN (DGCNN).**
- Converts each point into a feature vector.

📌 **Mathematical Operation:**  
\[
F = \text{DGCNN}(X)
\]
📌 **Shape Transformation:**  
\[
X \in \mathbb{R}^{B \times N \times 3} \quad \Rightarrow \quad F \in \mathbb{R}^{B \times N \times d}
\]
where \( d \) is the feature dimension.


### **2️⃣ Positional Embedding**
- **Encodes spatial information into the feature space.**

📌 **Mathematical Operation:**  
\[
F' = F + \text{PositionalEmbedding}(X)
\]
📌 **Shape:**  
\[
F \in \mathbb{R}^{B \times N \times d} \quad \Rightarrow \quad F' \in \mathbb{R}^{B \times N \times d}
\]
(Positional encoding does not change dimensions.)

### **3️⃣ Geometry-Aware Transformer Encoder**
- **Self-Attention applied to point proxies.**
- Models local and global geometric relations.

📌 **Mathematical Operation:**  
\[
V = \text{Encoder}(F')
\]
📌 **Shape:**  
\[
F' \in \mathbb{R}^{B \times N \times d} \quad \Rightarrow \quad V \in \mathbb{R}^{B \times N \times d}
\]

### **4️⃣ Query Generator (Dynamic Queries)**
- **Generates queries based on encoder output.**
- Queries represent missing point proxies.

📌 **Mathematical Operation:**  
\[
Q = \text{QueryGenerator}(V)
\]
📌 **Shape:**  
\[
V \in \mathbb{R}^{B \times N \times d} \quad \Rightarrow \quad Q \in \mathbb{R}^{B \times M \times d}
\]
where \( M \) is the number of missing point proxies.

### **5️⃣ Geometry-Aware Transformer Decoder**
- **Cross-attention between encoder outputs and queries.**
- Generates refined point proxy features.

📌 **Mathematical Operation:**  
\[
H = \text{Decoder}(Q, V)
\]
📌 **Shape:**  
\[
Q \in \mathbb{R}^{B \times M \times d}, \quad V \in \mathbb{R}^{B \times N \times d} \quad \Rightarrow \quad H \in \mathbb{R}^{B \times M \times d}
\]

### **6️⃣ FoldingNet (Final Point Cloud Generation)**
- **Generates fine-grained point clouds from point proxies.**
- Uses an MLP-based upsampling strategy.

📌 **Mathematical Operation:**  
\[
P = \text{FoldingNet}(H)
\]
📌 **Shape:**  
\[
H \in \mathbb{R}^{B \times M \times d} \quad \Rightarrow \quad P \in \mathbb{R}^{B \times M' \times 3}
\]
where \( M' \) is the number of final generated points.

## **📝 Summary Table (Shapes Per Module)**

| **Module**                            | **Input Shape**              | **Output Shape**            |
|----------------------------------------|------------------------------|-----------------------------|
| **1. Feature Extractor (DGCNN)**       | \( B \times N \times 3 \)     | \( B \times N \times d_{lf} \)    |
| **2. Positional Embedding**            | \( B \times N \times 3 \)     | \( B \times N \times d_{pe} \)    |
| **3. Transformer Encoder**             | \( B \times N \times d \)     | \( B \times N \times d \)    |
| **4. Query Generator**                 | \( B \times N \times d \)     | \( B \times M \times d \), \( B \times M \times 3 \)    |
| **5. Transformer Decoder (Cross-Attn)**| \( B \times M \times d_k \), \( B \times N \times d_v \) | \( B \times M \times d \) |
| **6. FoldingNet (Final Output)**       | \( B \times M \times 3 \)     | \( B \times M \times 3 \)  |

## **🚀 Final Output**
✅ **Complete 3D Point Cloud**:  
\[
P \in \mathbb{R}^{B \times M' \times 3}
\]
where \( M' \) is the number of reconstructed points.

This provides the **abstract mathematical operations and shape transformations** for each module in **Our Architecture**.🚀