> **Note:**
> Due to output rendering limitations on GitHub, some notebook cells (especially those involving dynamic outputs or widgets) may not display properly.
> To view the full outputs and run the notebook interactively, please open it in Google Colab:
> [explained-model.ipynb](https://colab.research.google.com/drive/12F-tvR6c8qm58_gMmKTi8QgnyknomM5t?usp=sharing)
> [biner-and-multiclass-classification.ipynb](https://colab.research.google.com/drive/1ASxKgCLiBljSqaQiz8AuI7kZc1rT-H31?usp=sharing)

# X-IDS: Explainable Intrusion Detection System

**X-IDS** is an explainable intrusion detection system that demonstrates how to build multi-stage cyber threat detection with human-readable outputs. This release focuses on data preprocessing and model training, serving as a foundation for future real-time deployment. The system integrates the strengths of **autoencoder-based anomaly detection**, **gradient boosting classification**, and **T5-small text generation** to deliver both accurate attack detection and transparent, natural language explanations.

---

## System Overview

The X-IDS system operates as a multi-stage decision pipeline with built-in logic to route traffic data through appropriate models. Each stage serves a distinct purpose and makes branching decisions based on the outcome of the previous model.

![System Process](asset/Network%20Feature.jpg)

### Step-by-Step Decision Flow:

1. **Anomaly Detection (Autoencoder - Step 1)**  
   - Input traffic data is first passed through a neural autoencoder trained on normal traffic.
   - If the reconstruction error is high (i.e., anomaly detected), the input is forwarded to the binary classification model (Model 1).
   - If the reconstruction error is low (i.e., normal traffic), the input is directly passed to the text generation model to produce a natural language explanation stating that the traffic is normal.

2. **Binary Classification (LightGBM - Step 2)**  
   - This model determines whether the anomalous input is likely to be a known attack or not.
   - If it predicts "attack", the input is forwarded to the attack type classifier (Model 2).
   - If it fails to classify confidently as attack, the sample is flagged as a potential novel anomaly and should be reviewed by a SOC (Security Operations Center) analyst for further investigation.

3. **Multi Class Classification (Multiclass LightGBM - Step 3)**  
   - This model attempts to classify the input into one of the 9 known attack categories.
   - If the input matches one of the known classes with sufficient confidence, it is passed to the text generation model to explain the detected attack.
   - If the prediction confidence is abnormally low, even though it technically falls into an existing class, it is flagged as a candidate for new attack patterns, and forwarded to human analysts for confirmation.

4. **Explanation Text Generation (T5-small - Step 4)**  
   - The final stage uses a fine-tuned T5-small model to generate a natural language explanation of the traffic behavior.
   - Depending on the routing, the model generates either:
     - A benign message (e.g., `"Trafik normal terdeteksi dengan durasi 0.292828 detik,..."`)
     - Or an attack explanation (e.g., `"Serangan generic terdeteksi dengan durasi..."`)

---

## Model Descriptions

### 1. Autoencoder (PyTorch)

- **Goal**: Unsupervised anomaly detection
- **Architecture**: Fully connected encoder–decoder
- **Thresholding**: Based on reconstruction error
- **Metric**: Recall and Confusion Matrix

**Recall (Positive Class / Anomaly)**: `1.0000`

> A perfect recall means **all positive (anomalous) instances were correctly identified**, which is crucial in domains like anomaly detection, fraud prevention, or medical diagnosis.

**Confusion Matrix:**
```
           Predicted
         |   0   |   1
     ------------------
Actual 0 | 93803 | 4900
Actual 1 |     0 | 4882
```

- **True Negatives (TN)**: 93,803  
- **False Positives (FP)**: 4,900  
- **False Negatives (FN)**: 0  
- **True Positives (TP)**: 4,882  

> The model **did not miss any anomaly** (Recall = 1.0).
> It had **4,900 false positives**, which can be acceptable depending on the application.

### 2. LightGBM Binary Classifier

- **Input**: Normalized features
- **Target**: Binary `label` (0 = normal, 1 = attack)
- **Metric**: ROC-AUC, Recall, F1-Score

| Metric     | Training | Testing |
|------------|----------|--------|
| **F1-score** | 0.9902 | 0.9300 |
| **Recall**   | 1.0000 | 0.9647 |
| **ROC-AUC**  | 1.0000 | 0.9994 |

### 3. LightGBM Multiclass Classifier

- **Input**: Same feature space
- **Target**: `attack_cat` with 9 classes
- **Metric**: Classification report

**Training Evaluation**:
```
          precision    recall  f1-score   support

       0       0.70      0.69      0.69     25014
       1       0.63      0.84      0.72     25014
       2       0.97      0.91      0.94     25014
       3       1.00      0.99      0.99     25014
       4       0.99      0.97      0.98     25014
       5       0.99      0.92      0.95     25014
       6       1.00      0.98      0.99     25014
       7       1.00      1.00      1.00     25014
       8       1.00      1.00      1.00     25014

accuracy                           0.89    200112
macro avg      0.91      0.89      0.90    200112
weighted avg   0.91      0.89      0.90    200112
```

**Testing Evaluation**:
```
          precision    recall  f1-score   support

       0       0.70      0.69      0.69     25014
       1       0.63      0.84      0.72     25014
       2       0.97      0.91      0.94     25014
       3       1.00      0.99      0.99     25014
       4       0.99      0.97      0.98     25014
       5       0.99      0.92      0.95     25014
       6       1.00      0.98      0.99     25014
       7       1.00      1.00      1.00     25014
       8       1.00      1.00      1.00     25014

accuracy                           0.89    200112
macro avg      0.91      0.89      0.90    200112
weighted avg   0.91      0.89      0.90    200112
```

### 4. T5-small Generative Explanation

- **Input Format**: Scaled features joined as text
- **Target**: Textual explanation
- **Training**: Hugging Face Trainer

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note for torch version:**  
> The `requirements.txt` file specifies `torch==2.6.0`, which may install a CPU-only version or a default CUDA build from PyPI.  
>  
> Since the `explained_model.ipynb` relies on GPU acceleration for efficient training,  
> it's recommended to install PyTorch with the `cu124` version manually: 
>  
> ```bash
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> ```  
>  
> This ensures full compatibility with your CUDA environment and enables GPU acceleration,  
> as `explained_model.ipynb` was executed on an A100 GPU.

---

## Dataset

- **Data Source**: [UNSW-NB15 on Kaggle](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)
- **Processed Dataset**: [`luminolous/xids-dataset`](https://huggingface.co/datasets/luminolous/xids-dataset)

---

## Models

For the models, you can check them on Hugging Face:
[`luminolous/xids-multimodel`](https://huggingface.co/luminolous/xids-multimodel)
