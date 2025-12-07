# Solar Power Prediction from Tabular Meteorological & Solar Geometry Data

## 1. Project Overview

### 1.1 Problem Statement

Solar energy generation depends heavily on environmental and solar-geometry variables. Accurate forecasting is essential for:

* Grid stability
* Battery scheduling and load balancing
* Energy market bidding
* Solar farm operation planning

The objective of this project is to develop and evaluate regression and neural network models that can predict solar power output from a wide range of meteorological and solar-position features. The final models aim to minimise forecasting error and improve prediction stability across varying conditions.

### 1.2 Data Summary

The dataset consists of time-indexed observations containing:

#### **Meteorological variables**

* Temperature (2 m above ground)
* Relative humidity
* Pressure
* Wind speed
* Cloud cover

#### **Solar radiation and geometry**

* Shortwave radiation
* Angle of incidence
* Zenith
* Azimuth

#### **Target variable**

* PV power output (in watts)

Several ANN architectures and an MLR baseline are trained and evaluated using a 70%/30% train–validation split after preprocessing, scaling, and feature refinement.

---

## 2. Repository Structure

A recommended structure for this project is:

```text
.
├── data/
│   ├── raw/
│   │   └── Task_1_Dataset_Solar_Power.csv
│   └── processed/
│       ├── solar_train.csv
│       └── solar_valid.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   └── 02_modeling_experiments.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── best_ann_model.h5
│
├── results/
│   ├── figures/
│   └── metrics_summary.csv
│
├── docs/
│   └── Solar_Power_Prediction_Report.pdf
│
├── README.md
└── requirements.txt
```

**Highlights**

* `data/` contains raw and processed datasets.
* `notebooks/` includes reproducible experimentation workflows.
* `src/` houses modular preprocessing, modeling, and evaluation code.
* `models/` stores trained ANN model weights.
* `results/` contains plots and metric summaries.

---

## 3. Installation & Requirements

### 3.1 Dependencies

Key libraries used:

* `numpy`, `pandas` – numerical data handling
* `matplotlib`, `seaborn` – exploratory visualisation
* `scikit-learn` – scaling, regression, and metrics
* `tensorflow` / `keras` – training ANN models

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
```

### 3.2 Setup

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>

pip install -r requirements.txt
```

Place the dataset file into:

```
data/raw/Task_1_Dataset_Solar_Power.csv
```

---

## 4. End-to-End Pipeline

This section outlines the complete workflow used to build the solar power prediction models.

---

### 4.1 Data Loading & Initial Inspection

1. **Load raw CSV**

```python
import pandas as pd
data = pd.read_csv("data/raw/Task_1_Dataset_Solar_Power.csv")
```

2. **Explore data quality**

* Check datatypes and missing values with `data.info()`.
* Analyse feature distributions with `data.describe()`.
* Identify any anomalies, constant columns, or unrealistic values.

3. **Visual exploration**

* Pairplots and histograms to understand feature behaviour
* Correlation matrix to identify most influential predictors
* Time-series plots for solar output and key environmental variables

---

### 4.2 Preprocessing & Feature Engineering

#### 4.2.1 Missing Value Handling

* Mean imputation is used for any numerical gaps.
* Ensures consistent dataset size for neural network training.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)
```

#### 4.2.2 Scaling

Neural networks and regression models benefit from normalized features.

`MinMaxScaler` scales all numerical values to `[0, 1]`:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_imputed)
```

#### 4.2.3 Train–Validation Split

A 70/30 split ensures sufficient data for ANN training:

```python
from sklearn.model_selection import train_test_split

train, valid = train_test_split(data_scaled, test_size=0.3, random_state=42)
```

---

### 4.3 Baseline Model: Multiple Linear Regression

A linear model is trained on carefully selected features such as:

* Temperature
* Relative humidity
* Shortwave radiation
* Angle of incidence
* Zenith
* Azimuth

This provides a benchmark for evaluating the benefit of nonlinear neural networks.

**Baseline performance:**

* **MAE:** ~420
* **RMSE:** ~544
* **R:** ~0.81

These results highlight the limitations of linear models for capturing nonlinear interactions in solar power generation.

---

### 4.4 ANN Model Development

Multiple ANN architectures were tested by varying:

* Depth (number of hidden layers)
* Width (neurons per layer)
* Activation functions (ReLU, Tanh)
* Optimizers (Adam, Nadam, RMSprop)
* Dropout rates
* Learning rates

#### Best-performing neural networks:

---

### **ANN11 — High-capacity deep model**

* **Architecture:** 128 → 96 → 64 → 32
* **Activation:** ReLU
* **Optimizer:** Adam
* **Performance:**

  * MAE ≈ 290
  * RMSE ≈ 431
  * R ≈ 0.887

This model extracts nonlinear relationships effectively and produces stable predictions.

---

### **ANN12 — Compact model with regularisation**

* **Architecture:** 64 → 32
* **Activation:** Tanh
* **Dropout:** Yes
* **Optimizer:** Nadam
* **Performance:**

  * MAE ≈ 294
  * RMSE ≈ 438
  * R ≈ 0.888

This model generalizes well while keeping computational cost low.

---

### 4.5 Training Procedure

Typical training setup:

* Optimizer: Adam or Nadam
* Loss: Mean Squared Error (MSE)
* Batch size: 64
* Epochs: up to 100
* Early stopping to avoid overfitting

Example Keras workflow:

```python
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=64
)
```

---

### 4.6 Model Evaluation & Visualisation

Evaluation metrics:

* **MAE** (Mean Absolute Error)
* **RMSE** (Root Mean Square Error)
* **R** (Correlation coefficient)
* **R²** (Coefficient of determination)

Visualisations include:

* True vs predicted scatter plots
* Training/validation loss curves
* Error distribution histograms
* Residual analysis

ANN models show strong clustering around the identity line and outperform the MLR baseline by a clear margin.

---

## 5. Key Insights & Learnings

* Nonlinear ANNs significantly outperform linear regression for solar power prediction.
* Solar geometry variables (angle of incidence, zenith, azimuth) are powerful predictors.
* Deeper networks (ANN11) capture complex relationships, while smaller networks (ANN12) generalize efficiently.
* Proper scaling and imputation are essential for stable ANN training.
* ANN models reduce error by approximately **30%** relative to MLR.
* Some variability remains due to rapid weather fluctuations — future work may use LSTM/transformer hybrids or ensembles.

---

## 6. Real-world Applications

This modeling pipeline can support:

* **Solar farm operators** for generation scheduling
* **Utility providers** for grid balancing
* **Energy traders** for bidding strategies
* **Smart homes & microgrids** for battery optimisation
* **Research labs** exploring hybrid renewable forecasting

Future enhancements could incorporate real-time weather APIs, cloud movement models, or spatiotemporal forecasting.

---

## 7. How to Run This Project

1. Clone the repository
2. Install dependencies using `requirements.txt.`
3. Place your CSV dataset into `data/raw/`
4. Run preprocessing via:

   ```bash
   python src/data_preprocessing.py
   ```
5. Train models using:

   ```bash
   python src/train.py
   ```
6. Evaluate models and generate plots using:

   ```bash
   python src/evaluate.py
   ```
7. Optionally run the Jupyter notebooks for an interactive exploration of the entire pipeline.

## Follow me for more content related to AI and Data Analytics on
LinkedIn: https://www.linkedin.com/in/jasonpham808/
Medium: https://medium.com/@jasonpham808
