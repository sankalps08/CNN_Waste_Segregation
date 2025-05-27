# 🧠 Waste Classification using CNN

This project implements a custom **Convolutional Neural Network (CNN)** to classify waste images into 7 categories. It supports **automated waste segregation** — a critical component in smart city and sustainable development initiatives.

---

## 📂 Dataset Overview

The dataset consists of colored images categorized into:

- **Cardboard**
- **Food_Waste**
- **Glass**
- **Metal**
- **Other**
- **Paper**
- **Plastic**

Each image was resized to a consistent dimension of **128×128×3** for uniform input to the CNN model.

---

## 📊 Class Distribution

The dataset is **imbalanced**:

- **Plastic** is the majority class  
- **Cardboard** is underrepresented  

**Risks:**
- Skewed model performance
- Bias toward frequently occurring classes

**Mitigations:**
- Applied **data augmentation**
- Consider future **class weighting** and **oversampling techniques**

---

## 🛠️ Preprocessing

- Images normalized to **[0, 1]**
- Labels extracted from folder names
- Applied **one-hot encoding** for multi-class classification
- **Stratified split** into training and validation sets (70:30)

---

## 🧠 Model Architecture

Built using **TensorFlow/Keras**, the CNN includes:

- **3 Convolutional blocks**:  
  Each: `Conv2D → BatchNorm → ReLU → MaxPool → Dropout`

- **Dense Layers**:  
  After flattening, used `Dense(512)` → Dropout(0.5), `Dense(256)` → Dropout(0.5)

- **Output Layer**:  
  `Dense(7, activation='softmax')` for classification

### 🔧 Design Choices

| Component           | Justification |
|---------------------|---------------|
| **BatchNorm**       | Faster and stable training |
| **ReLU Activation** | Avoids vanishing gradients |
| **Dropout**         | Reduces overfitting |
| **Softmax**         | Multiclass classification |

---

## ⚙️ Training Configuration

- **Loss Function**: `CategoricalCrossentropy`
- **Optimizer**: `Adam` with `learning_rate = 0.005`
- **Callbacks**:
  - `EarlyStopping(patience=8)`
  - `ModelCheckpoint`
  - `ReduceLROnPlateau`

---

## 🧪 Data Augmentation

To combat overfitting and improve minority class performance:

```python
ImageDataGenerator(
    rotation_range=7,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)


## 📈 Results and Evaluation

### 🔹 Confusion Matrix (Before vs After Augmentation)

- **Before**: Model overfit on majority classes such as **Plastic**.
- **After**: Greater diagonal dominance, especially in underrepresented classes like **Cardboard** and **Glass**, indicating better classification accuracy.

---

### 🔹 Classification Report (After Augmentation)

| **Metric**   | **Validation Result**       |
|--------------|-----------------------------|
| **Accuracy** | ~68%                        |
| **Precision**| Improved across all classes |
| **Recall**   | Higher for minority classes |
| **F1-Score** | Balanced (~0.68 macro avg)  |

---

## 🎯 Business Impact

- Enables **automated waste classification** in real-time environments.
- Reduces **manual sorting errors and labor costs**.
- Easily scalable for **smart city infrastructure and recycling centers**.

---

## 🔮 Future Work

- Apply **class weighting** or **SMOTE** to improve learning on imbalanced classes.
- Integrate **transfer learning** (e.g., MobileNet, ResNet) to leverage pretrained visual features.
- Experiment with alternate activations like **LeakyReLU** or **ELU** for deeper learning performance.
- Use **GANs** or collect more data to boost performance on rare waste categories.

---
