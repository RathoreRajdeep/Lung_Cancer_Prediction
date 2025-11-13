# Lung Cancer Classification using CNN Ensemble + Reinforcement Learning

This project presents an advanced **lung cancer classification system** that uses an **ensemble of five pre-trained CNN models**, combined through a **Reinforcement Learning (Q-learning) agent** that dynamically learns optimal model weights for improved prediction accuracy.

The system classifies CT lung images into:

- **Benign**
- **Malignant**
- **Normal**

This project aligns strongly with healthcare AI, ensemble modeling, and agent-based ML workflows.

---

## 🚀 Project Highlights

### ✔ 5 Pretrained CNN Models Used  
- DenseNet201  
- EfficientNetB7  
- VGG16  
- VGG19  
- MobileNet  

Each model is trained separately using transfer learning on the **IQ-OTHNCCD Lung Cancer Dataset**.

---

## 🎯 Reinforcement Learning Ensemble

Instead of static averaging, an **RL agent** dynamically updates weight contributions of each model based on:

- Individual model accuracy  
- Reward generated from improvement  
- Exploration–exploitation strategy  
- Experience replay  

This makes the ensemble **self-improving**.

### 📈 Example Weight Evolution  
(Place your file inside results/weight_changes.png)
```
![Weight Changes](results/weight_changes.png)
```

---

## 📂 Dataset

**IQ-OTHNCCD Lung Cancer Dataset**  
Kaggle link: https://www.kaggle.com/datasets

- **878** training images  
- **219** validation images  
- **3 classes**: Benign, Malignant, Normal  

Preprocessing used:

- Resize to 224×224  
- Normalization  
- Data augmentation  
- Batch generator  

---

## 🧠 Project Architecture

(Place your file inside assets/architecture.png)
```
![Architecture Diagram](assets/architecture.png)
```

---

## 🔧 Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Scikit-learn  
- Reinforcement Learning (Q-learning)  

---

## 🏗 Folder Structure

```
lung-cancer-ensemble-RL/
│
├── lung_cancer_project.ipynb
├── README.md
├── requirements.txt
│
├── models/
│     ├── MobileNet_model.h5
│     ├── VGG16_model.h5
│     ├── VGG19_model.h5
│     ├── DenseNet201_model.h5
│     └── EfficientNetB7_model.txt   # Link to large model
│
├── results/
│     └── weight_changes.png
│
└── assets/
      └── architecture.png
```

---

## 🔍 Model Training Results

| Model            | Train Acc | Val Acc |
|-----------------|-----------|---------|
| DenseNet201      | 94.66%    | 79.00% |
| EfficientNetB7   | 48%       | 51%    |
| VGG16            | 84.97%    | 81.74% |
| MobileNet        | 97.18%    | 86.76% |
| VGG19            | 82.07%    | 76.71% |

The RL agent learns to **assign more weight to stronger models** (MobileNet, VGG16, DenseNet201).

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

---

## ▶️ Run Inference

```
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("models/VGG16_model.h5")

img = Image.open("test.jpg").resize((224,224))
img = np.expand_dims(np.array(img) / 255.0, axis=0)

pred = model.predict(img)
print(pred)
```

---

## 🌟 Future Improvements

- Add Grad-CAM explainability  
- Deploy using AWS Lambda / FastAPI  
- Convert system into an agentic radiology assistant  
- Add LLM-based medical report generation  

---

## 📜 License
MIT License

---

## 👤 Author  
**Rajdeep Singh Rathore**

Feel free to open issues or contribute!
"# Lung_Cancer_Prediction" 
