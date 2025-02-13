# CrownGeneration
AI Pipeline for Crown Generation Module

## **📌 Our Architecture**
![OurArchitecture](/images/Pipeline.png)

## **📌 Abstract Representation of our Architecture**
Our Architecture is a **transformer-based** encoder-decoder model for **point cloud construction**.

1. **Input:**  
   - **point cloud** → `X ∈ R^(B × N × 3)`  
   - Where `B` = batch size, `N` = number of points, and **each point has 3D coordinates (x, y, z)**.  

### **1️⃣ Feature Extraction (DGCNN)**
- **Extracts local features using a Dynamic Graph CNN (DGCNN).**
- Converts each point into a feature vector.

📌 **Mathematical Operation:**  
`F = DGCNN(X)`

📌 **Shape Transformation:**  
`X ∈ R^(B × N × 3) → F ∈ R^(B × N × d)`

where `d` is the feature dimension.

### **2️⃣ Positional Embedding**
- **Encodes spatial information into the feature space.**

📌 **Mathematical Operation:**  
`F' = F + PositionalEmbedding(X)`

📌 **Shape:**  
`F ∈ R^(B × N × d) → F' ∈ R^(B × N × d)`

(Positional encoding does not change dimensions.)

### **3️⃣ Geometry-Aware Transformer Encoder**
- **Self-Attention applied to point proxies.**
- Models local and global geometric relations.

📌 **Mathematical Operation:**  
`V = Encoder(F')`

📌 **Shape:**  
`F' ∈ R^(B × N × d) → V ∈ R^(B × N × d)`

### **4️⃣ Query Generator (Dynamic Queries)**
- **Generates queries based on encoder output.**
- Queries represent missing point proxies.

📌 **Mathematical Operation:**  
`Q = QueryGenerator(V)`

📌 **Shape:**  
`V ∈ R^(B × N × d) → Q ∈ R^(B × M × d)`

where `M` is the number of missing point proxies.

### **5️⃣ Geometry-Aware Transformer Decoder**
- **Cross-attention between encoder outputs and queries.**
- Generates refined point proxy features.

📌 **Mathematical Operation:**  
`H = Decoder(Q, V)`

📌 **Shape:**  
`Q ∈ R^(B × M × d), V ∈ R^(B × N × d) → H ∈ R^(B × M × d)`

### **6️⃣ FoldingNet (Final Point Cloud Generation)**
- **Generates fine-grained point clouds from point proxies.**
- Uses an MLP-based upsampling strategy.

📌 **Mathematical Operation:**  
`P = FoldingNet(H)`

📌 **Shape:**  
`H ∈ R^(B × M × d) → P ∈ R^(B × M' × 3)`

where `M'` is the number of final generated points.

## **📝 Summary Table (Shapes Per Module)**

| **Module**                            | **Input Shape**              | **Output Shape**            |
|----------------------------------------|------------------------------|-----------------------------|
| **1. Feature Extractor (DGCNN)**       | `B × N × 3`                  | `B × N × d_lf`             |
| **2. Positional Embedding**            | `B × N × 3`                  | `B × N × d_pe`             |
| **3. Transformer Encoder**             | `B × N × d`                  | `B × N × d`                |
| **4. Query Generator**                 | `B × N × d`                  | `B × M × d`, `B × M × 3`   |
| **5. Transformer Decoder (Cross-Attn)**| `B × M × d_k`, `B × N × d_v` | `B × M × d`                |
| **6. FoldingNet (Final Output)**       | `B × M × 3`                  | `B × M × 3`                |

## **🚀 Final Output**
✅ **Constructed 3D Point Cloud**:  
`P ∈ R^(B × M × 3)`

where `M` is the number of reconstructed points.

This provides the **abstract mathematical operations and shape transformations** for each module in **Our Architecture**.🚀