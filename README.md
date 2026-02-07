# Forecasting Next-Year Crop Yields Using a Multilayer Perceptron (MLP)

This project focuses on forecasting **next-year crop yields** using a **Multilayer Perceptron (MLP)** neural network built with **TensorFlow/Keras**. The model learns from historical agricultural data to predict future crop yields for specific **country–crop combinations**, supporting data-driven decision-making in agriculture and food security planning.

---

## Project Overview

Accurate crop yield forecasting is critical for agricultural planning, supply chain management, and policy decisions. This project formulates crop yield prediction as a **supervised regression problem**, where historical yield, temporal information, and categorical context are used to predict yield one year ahead.

The solution emphasizes:
- Robust preprocessing and feature engineering  
- Time-aware label construction  
- Regularized neural network modelling  
- Quantitative evaluation using standard regression metrics  

---

## Dataset Description

- **Total records:** 67,963 observations  
- **Granularity:** (country, crop, year)  
- **Time range:** 2010 – 2022  
- **Target variable:** `yield_next` (crop yield at year *t+1*)

### Input Features
- `year` – calendar year  
- `yield` – crop yield at year *t* (lag feature)  
- `country` – country of production (categorical)  
- `crop` – crop type (categorical)  

Categorical variables are encoded using **OneHotEncoder**, and numerical features are standardized using **StandardScaler**.

---

## Methodology

### Preprocessing
- Integrated yield, environmental, land-cover, and country metadata  
- Aggregated monthly environmental variables into annual values  
- Constructed next-year target labels using grouped time shifts  
- Removed incomplete records to ensure label consistency  
- Applied feature scaling and encoding via `ColumnTransformer`

### Model Architecture
- **Model:** Keras Sequential MLP  
- **Hidden layers:**  
  - 128 neurons (ReLU) + Dropout (0.3)  
  - 64 neurons (ReLU) + Dropout (0.2)  
- **Output layer:** 1 neuron (linear activation)  
- **Loss function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  

### Training Strategy
- Train–validation split: **80:20** (`random_state=42`)  
- Epochs: **100**  
- Batch size: **64**  
- Overfitting control: Dropout, feature scaling, validation monitoring  

---

## Model Performance

The trained MLP demonstrates strong predictive capability across the dataset:

- **MAE:** 1058.19  
- **RMSE:** 3531.24  
- **R² Score:** 0.9406  

The high R² value indicates that the model explains over **94% of the variance** in next-year crop yield, confirming robustness and generalisation across crops and regions.

---

## Repository Structure

