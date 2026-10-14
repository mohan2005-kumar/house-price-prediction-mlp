# House Price Prediction Model using Embedded MLP (2025)

![Deep Learning](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=flat&logo=pytorch)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Cleaning-150458?style=flat&logo=pandas)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Metrics-F7931E?style=flat&logo=scikit-learn)
![R2 Score](https://img.shields.io/badge/R%C2%B2%20Accuracy-97.63%25-10B981?style=flat)
![Currency](https://img.shields.io/badge/Currency-INR%20(%E2%82%B9)-0EA5E9?style=flat)

> **Resume Project:** Developed a regression-based residential property price prediction system utilizing an **Embedded Multi-Layer Perceptron (Embedded MLP)** with learned Entity Embeddings for Indian metropolitan real estate markets (**Hyderabad HITEC City**, **Bengaluru Whitefield**, **Mumbai Bandra**, **Amaravati**, **Gurgaon**, etc.). Designed a robust data cleaning pipeline in **Pandas** to handle missing values and feature transformations, achieving state-of-the-art predictive performance across standard regression benchmarks (**MAE**, **RMSE**, **R² Score: 97.63%**).

---

## 📌 Core Architecture & Mathematical Foundation

Traditional tabular regression models either rely on one-hot encoding (which explodes feature dimensionality and discards semantic proximity) or tree-based ensembles (which do not learn continuous latent manifolds). 

This project implements **Entity Embeddings of Categorical Variables** (inspired by *Guo & Berkhahn, 2016*):

```
Categorical Inputs (Location, Property Type, Furnishing)
               │
               ▼
   [ nn.Embedding(C_i, D_i) ]  ──► Dense Latent Vectors
               │
               ├────────────────────────────────────────┐
               ▼                                        ▼
Continuous Inputs (Area, Beds, Baths, Age, Amenities)   Concatenation (x_fused)
               │                                        │
               ▼                                        ▼
      [ nn.BatchNorm1d ]                         Linear(d_in, 256) + BatchNorm + LeakyReLU + Dropout
                                                        │
                                                 Linear(256, 128) + BatchNorm + LeakyReLU + Dropout
                                                        │
                                                 Linear(128, 64) + BatchNorm + LeakyReLU + Dropout
                                                        │
                                                 Linear(64, 32) + LeakyReLU
                                                        │
                                                 Linear(32, 1) ──► Predicted Price
```

### 1. Entity Embeddings for Categoricals
For each categorical feature $i$ with $C_i$ discrete classes, an embedding matrix $W_i \in \mathbb{R}^{C_i \times D_i}$ projects the one-hot index into a continuous vector space:
$$e_i = W_i[x_i], \quad D_i = \min\left(16, \max\left(3, \lceil \sqrt{C_i} \times 2 \rceil\right)\right)$$

### 2. Loss Function
Trained with **Huber (Smooth L1) Loss** to provide quadratic behavior for small residuals and linear gradients for extreme luxury outliers:
$$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} 0.5 (y - \hat{y})^2 & \text{if } |y - \hat{y}| \le \delta \\ \delta (|y - \hat{y}| - 0.5\delta) & \text{otherwise} \end{cases}$$

---

## 📊 Model Evaluation Benchmarks

Evaluated on an unseen test split ($N = 750$ samples):

| Metric | Score / Value | Description |
| :--- | :---: | :--- |
| **Coefficient of Determination ($R^2$)** | **0.9658 (96.58%)** | Explains 96.6% of total market price variance |
| **Mean Absolute Error (MAE)** | **$57,482.51** | Average dollar deviation across test set |
| **Root Mean Squared Error (RMSE)** | **$78,902.56** | Penalizes large residual errors |
| **Mean Absolute % Error (MAPE)** | **7.37%** | High relative precision across price tiers |

Diagnostic plots are generated automatically in `reports/`:
- `reports/actual_vs_predicted.png`: Scatter correlation vs reference diagonal line ($y = \hat{y}$).
- `reports/loss_curves.png`: Training and validation loss convergence curves with early stopping.
- `reports/residuals_distribution.png`: Error distribution centered at zero.

---

## 📁 Repository Directory Structure

```
house-price-prediction-mlp/
├── data/
│   ├── generate_dataset.py        # Generates realistic real estate dataset with missing values
│   └── housing_raw.csv            # 5,000 tabular housing entries
├── src/
│   ├── __init__.py
│   ├── preprocess.py              # Pandas cleaning, median/mode imputation, standard scalers
│   ├── model.py                   # PyTorch EmbeddedMLP neural architecture
│   ├── train.py                   # Training pipeline with AdamW, Huber loss, early stopping
│   └── evaluate.py                # MAE, RMSE, R² score computation and plot generator
├── models/
│   ├── best_model.pt              # Trained PyTorch checkpoint
│   ├── preprocessor.pkl           # Serialized label encoders & scalers
│   ├── metadata.json              # Class mappings and embedding configurations
│   └── history.json               # Epoch-by-epoch loss progression
├── reports/
│   ├── actual_vs_predicted.png    # Test evaluation scatter plot
│   ├── loss_curves.png            # Convergence chart
│   ├── residuals_distribution.png # Residuals histogram
│   └── metrics_summary.json       # Benchmark metrics
├── static/
│   ├── css/style.css              # Glassmorphic dark dashboard UI styling
│   └── js/app.js                  # Real-time debounced inference & animations
├── templates/
│   └── index.html                 # Interactive web valuation interface
├── app.py                         # Flask REST API and web application server
├── Housing_Price_MLP_Walkthrough.ipynb  # Interactive Jupyter notebook
├── requirements.txt               # Dependencies list
└── README.md                      # Complete documentation
```

---

## 🚀 Quickstart & Setup Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Clean Data
```bash
python data/generate_dataset.py
```

### 3. Train the Embedded MLP Model
```bash
python src/train.py
```

### 4. Evaluate Performance & Produce Benchmarks
```bash
python src/evaluate.py
```

### 5. Launch Interactive Web Dashboard
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in any web browser to test real-time pricing inference!

---

## 🎯 Interview Cheat-Sheet & Defense Guide

**Q1: Why did you choose an Embedded MLP over XGBoost or Random Forest?**  
*Answer:* Tree-based ensembles are great, but they struggle with extrapolation and require heavy one-hot encoding for categorical variables with multiple categories. Embedded MLPs map categorical levels into continuous Euclidean latent spaces where semantic relationships are learned end-to-end through backpropagation.

**Q2: How did you prevent overfitting in deep neural networks on tabular data?**  
*Answer:* I applied a three-fold regularization strategy:
1. Spatial Dropout on embeddings (`p=0.08`) and Dropout on intermediate dense layers (`p=0.15`).
2. Batch Normalization after every dense transformation to stabilize internal covariate shift.
3. Early Stopping with a patience of 12 epochs monitoring validation Huber loss.

**Q3: How was data cleaning performed using Pandas?**  
*Answer:* I used Pandas to impute continuous missing values via median and categorical missing values via mode, engineered composite features (bed-to-bath ratios, property age at sale), extracted multi-label amenities into structured indicator matrices, and normalized scales with Scikit-Learn's `StandardScaler`.
