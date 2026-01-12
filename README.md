# Paper Reproduction and Extension: Anomaly Transformer

This repository is based on the official implementation of the paper:

Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy  
ICLR 2022  
Original repository: https://github.com/thuml/Anomaly-Transformer

---

## Purpose

The goal of this project is to reproduce the original results of the paper and further analyze and evaluate the proposed method through additional experiments and modifications.

This repository is created for academic and educational purposes.




## Experiments and Extensions

Based on the original implementation of Anomaly Transformer, I conducted two additional experimental studies on the SWaT dataset.

### Experiment 1 — Early Anomaly Detection

The original paper evaluates point-wise anomaly detection only.  
In this experiment, I evaluate the early-warning capability of Anomaly Transformer by defining an early detection horizon \(k \in \{5,10,20\}\) before the first labeled anomaly timestamp.

A detection is considered successful if the anomaly score exceeds the threshold within the horizon.

Metrics:
- Early Detection Rate (EDR)
- Average early lead time
- Standard F1 / Precision / Recall

---

### Experiment 2 — Robustness to Data Corruption

To evaluate robustness under realistic sensor conditions, I tested the trained model on corrupted SWaT test data:

- Additive Gaussian noise on selected channels.
- Random missing values with different masking ratios and simple imputations.
- Sensor spike simulation.

Metrics:
- F1 / Precision / Recall / ROC-AUC
- Relative performance drop compared to clean data.

---

These experiments aim to analyze the practical applicability and robustness of Anomaly Transformer beyond the original paper settings.
