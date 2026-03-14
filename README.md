# 🔩 Bearing Fault Detection — Predictive Maintenance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikit-learn)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![CI](https://img.shields.io/github/actions/workflow/status/yourusername/bearing-fault-detection/ci.yml?style=flat-square&label=CI)

**End-to-end predictive maintenance pipeline for rotating machinery, from raw vibration signals to real-time fault diagnosis.**

[Live Demo](#) · [Documentation](#architecture) · [Dataset](#dataset)

</div>

---

## 📌 Context & Motivation

Bearing failures account for **40–50% of all rotating machine breakdowns**, causing unplanned downtime and significant industrial losses. Early fault detection — before catastrophic failure — is the cornerstone of modern predictive maintenance strategies.

This project implements a **full ML pipeline** that:
1. Processes raw vibration signals (accelerometer data)
2. Extracts rich time-frequency features
3. Classifies bearing faults with >97% accuracy
4. Provides an interactive real-time diagnosis dashboard

> Built as an extension of a 3rd-place national hackathon project (MTI Hackathon 2025) on Physics-Informed Neural Networks for bearing degradation modeling.

---

## 🏗️ Architecture

```
bearing-fault-detection/
│
├── src/
│   ├── data/
│   │   ├── download_data.py       # CWRU dataset downloader & parser
│   │   └── preprocessing.py       # Signal windowing & normalization
│   │
│   ├── features/
│   │   └── feature_extraction.py  # Time + Frequency + Envelope features
│   │
│   ├── models/
│   │   ├── train.py               # Classical ML (XGBoost, Random Forest)
│   │   ├── pinn.py                # Physics-Informed Neural Network
│   │   └── evaluate.py            # Metrics, confusion matrix, SHAP
│   │
│   └── dashboard/
│       └── app.py                 # Streamlit interactive dashboard
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Model_Comparison.ipynb
│
├── tests/
│   ├── test_features.py
│   └── test_models.py
│
├── .github/workflows/
│   └── ci.yml                     # Automated testing & linting
│
├── config.yaml                    # Centralized configuration
└── requirements.txt
```

---

## 🎯 Fault Classes

| Class | Label | Description | Severity |
|-------|-------|-------------|----------|
| Normal | `NOR` | Healthy bearing | — |
| Inner Race Fault | `IRF` | Defect on inner ring | 0.007" / 0.014" / 0.021" |
| Outer Race Fault | `ORF` | Defect on outer ring | 0.007" / 0.014" / 0.021" |
| Ball Fault | `BF` | Rolling element defect | 0.007" / 0.014" / 0.021" |

---

## 📊 Feature Engineering

The pipeline extracts **52 features** across three domains:

### Time Domain (18 features)
- RMS, Peak, Crest Factor, Kurtosis, Skewness
- Peak-to-Peak, Shape Factor, Impulse Factor, Clearance Factor
- Statistical moments (mean, std, variance)

### Frequency Domain (20 features)
- FFT Power Spectrum (dominant frequencies)
- Bearing Defect Frequencies: BPFI, BPFO, BSF, FTF
- Spectral centroid, bandwidth, rolloff

### Envelope Analysis (14 features)
- Hilbert Transform → Envelope signal
- Envelope spectrum peaks at fault characteristic frequencies
- Modulation sideband energy

---

## 🤖 Models

### 1. Classical ML Baseline
```
XGBoost Classifier
  └─ Accuracy: 97.8%
  └─ F1-Score (macro): 0.977
  └─ Inference: <1ms per sample

Random Forest
  └─ Accuracy: 96.4%
  └─ Best for: interpretability via feature importance
```

### 2. Physics-Informed Neural Network (PINN)
A custom PINN that incorporates bearing mechanics equations as physics constraints in the loss function:

```
L_total = L_data + λ · L_physics

L_physics = ||ω_fault_predicted - ω_fault_theoretical||²
```

Where theoretical fault frequencies are derived from bearing geometry (BPFI, BPFO, BSF, FTF formulas).

This ensures the model **respects physical laws** even in low-data regimes, making it more robust than pure data-driven approaches.

---

## 📈 Results

| Model | Accuracy | F1 (macro) | Training Time |
|-------|----------|------------|---------------|
| Random Forest | 96.4% | 0.961 | 12s |
| XGBoost | **97.8%** | **0.977** | 18s |
| PINN | 95.2% | 0.949 | 45s |
| SVM (baseline) | 89.1% | 0.887 | 8s |

*Evaluated on CWRU dataset, 12kHz drive-end accelerometer, 80/20 train-test split*

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/yourusername/bearing-fault-detection.git
cd bearing-fault-detection
pip install -r requirements.txt
```

### Download Dataset
```bash
python src/data/download_data.py
```

### Train Models
```bash
python src/models/train.py --model xgboost --save
```

### Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 📦 Dataset

This project uses the **[CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)** — the most widely used public benchmark for bearing fault diagnosis research.

- **Institution**: Case Western Reserve University (Ohio, USA)
- **Signal**: Accelerometer data at 12kHz and 48kHz
- **Loads**: 0, 1, 2, 3 HP motor loads
- **Setup**: Drive-end and fan-end bearing measurements

---

## 🔬 Scientific Background

Bearing fault frequencies are derived from geometry:

| Fault | Formula | Description |
|-------|---------|-------------|
| BPFI | `(N/2) · (1 + d/D · cos α) · fr` | Ball Pass Freq. Inner race |
| BPFO | `(N/2) · (1 - d/D · cos α) · fr` | Ball Pass Freq. Outer race |
| BSF | `(D/2d) · (1 - (d/D)² · cos²α) · fr` | Ball Spin Frequency |
| FTF | `(1/2) · (1 - d/D · cos α) · fr` | Fundamental Train Freq. |

Where `N` = number of balls, `d` = ball diameter, `D` = pitch diameter, `α` = contact angle, `fr` = shaft rotation frequency.

---

## 📚 References

- [1] Smith, W.A. & Randall, R.B. (2015). *Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study*. Mechanical Systems and Signal Processing.
- [2] Raissi, M. et al. (2019). *Physics-informed neural networks*. Journal of Computational Physics.
- [3] Lei, Y. et al. (2020). *Applications of machine learning to machine fault diagnosis*. Mechanical Systems and Signal Processing.

---

## 👤 Author

**Alaa Eddine TAHIR** — Engineering student at Arts et Métiers ParisTech  
[LinkedIn](https://linkedin.com/in/alaaeddine-tahir) · [GitHub](https://github.com/yourusername)

---

*This project extends work done at the MTI Hackathon 2025 (3rd place nationally) on PINN-based bearing degradation modeling.*
