# Generative-AI Course Project
## 🌡️ Kolkata 7‑Day Temperature Forecasts with TimeXer

> Daily maximum & minimum temperature predictions for Kolkata, India  
> **Horizon:** 7 days | **Data:** 1975‑01‑01 → 2025‑04‑08 | **Model:** TimeXer Transformer

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-TimeXer-ff69b4.svg)](https://github.com/unit8co/TimeX)

## 1. Project Overview
This repository contains code and notebooks to train and evaluate a **TimeXer** transformer for 7‑day ahead forecasting of daily **maximum** and **minimum** temperatures in Kolkata (22.57 °N, 88.36 °E).  
Key goals:

* Serve as a reference implementation for long‑horizon weather forecasting with TimeXer. 
* Provide reproducible benchmarks (RMSE, MAE, MAPE, R²) for each lead time (1 → 7 days).  
* Offer a lightweight, deployable model for downstream analytics or alerting pipelines.  


## 2. Data

| Source | Period | Frequency | Variables |
|--------|--------|-----------|-----------|
| [Open‑Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | 1975‑01‑01 → 2025‑04‑08 | Daily | `temperature_2m_max`, `temperature_2m_min` |



## 3. Model

```python
from timexer import TimeXer
from neuralforecast.losses.pytorch import MSE, MAE

model = TimeXer(
    h=7,             # forecasting horizon (days)
    input_size=30,   # history window length (days)
    n_series=1,
    batch_size=128,
    patch_len=12,
    hidden_size=128,
    n_heads=16,
    e_layers=2,
    d_ff=256,
    factor=1,
    dropout=0.2,
    use_norm=True,
    loss=MSE(),
    learning_rate=1e‑4,
    valid_loss=MAE(),
    early_stop_patience_steps=10,
    max_steps=2000
)
```

## 4. Training & Evaluation

| Split | Dates | Samples |
|-------|-------|---------|
| **Train** | 1975‑01‑01 → 2020‑12‑31 | 16 802 |
| **Test**  | 2021‑01‑01 → 2025‑04‑08 | 1 559  |

During testing, we **roll** the model forward one day at a time, producing a full 7‑day vector for each date. Metrics are computed **per lead‑time**:

```text
lead‑1 RMSE, MAE, MAPE, R²
lead‑2 RMSE, MAE, MAPE, R²
…
lead‑7 RMSE, MAE, MAPE, R²
```

## 6. Results

| Lead | RMSE (°C) | MAE (°C) | MAPE (%) | R² |
|------|---------:|---------:|---------:|----:|
| **Day 1** | 1.310 | 0.987 | 4.07 | 0.907 |
| **Day 2** | 1.700 | 1.292 | 5.38 | 0.844 |
| **Day 3** | 1.930 | 1.470 | 6.13 | 0.798 |
| **Day 4** | 2.088 | 1.593 | 6.62 | 0.763 |
| **Day 5** | 2.178 | 1.668 | 6.96 | 0.741 |
---

## 7. Re‑using the Model
.
```python
from timexer import TimeXer
import torch
.
model = TimeXer.load_from_checkpoint("artifacts/model.ckpt")
history = df.tail(30)               # last 30 days of max/min temps
forecast = model.predict(history)   # numpy array shape (7, 2)
```

---

## 8. Repository Structure

```
 .
 ├── data/
 │   ├── raw/                # cached API responses
 │   └── processed/          # cleaned parquet files
 ├── notebooks/
 │   ├── 01_data_download.ipynb
 │   ├── 02_model_training.ipynb
 │   └── 03_evaluation.ipynb
 ├── src/
 │   ├── download_data.py
 │   ├── train_model.py
 │   └── evaluate.py
 ├── artifacts/              # checkpoints, metrics, plots
 └── README.md
 ```

---

*Questions or ideas?* Open an issue or ping me on [mail](mailto:abhiroopsarkar2004@gmail.com).  
