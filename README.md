# Wafer-Map-Pattern-Intelligence
WaferAI: Embedded EfficientNet-B0 with Comprehensive Monte Carlo Yield Simulations for Real-Time 300mm Semiconductor Fab Defect Intelligence and Economic Optimization

# Wafer Map Pattern Intelligence

An **embedded edge-AI system for semiconductor wafer defect intelligence**, designed to classify wafer-map defect patterns, detect unknown anomalies, localize model decisions, and estimate yield impact in real time.

## Problem

Wafer defects reveal underlying fabrication-process failures, but conventional inspection workflows can be **manual, closed-set, and reactive**, making unknown defects and emerging process failures difficult to detect early.

## Solution

The system combines **EfficientNet-B0 deep feature extraction**, supervised defect classification, **Mahalanobis-based open-set recognition**, convolutional-autoencoder anomaly detection, **Grad-CAM explainability**, and yield/economic impact analysis.

## Architecture / Workflow

```text
Wafer Map
   ↓
Preprocessing
   ↓
EfficientNet-B0
   ↓
1280-D Feature Embedding
   ├──→ Defect Classification
   ├──→ Open-Set / OOD Detection
   └──→ Anomaly Detection
             ↓
        Grad-CAM
             ↓
   Yield & Economic Analysis
             ↓
   Process / Root-Cause Intelligence
```

## Embedded AI Deployment

The trained model is designed for **edge inference on embedded AI hardware** such as NVIDIA Jetson, FPGA accelerators, or Edge TPU platforms. INT8 quantization and hardware-accelerated inference can enable **low-latency, on-device wafer inspection** without requiring continuous cloud connectivity.

## Key Capabilities

* **EfficientNet-B0** — spatial defect representation learning
* **Open-Set Recognition** — detects previously unseen defect distributions
* **Anomaly Detection** — identifies abnormal wafer patterns without labels
* **Grad-CAM** — interpretable defect localization
* **Yield Intelligence** — estimates defect-driven yield and economic impact
* **Process Diagnostics** — maps defect patterns to potential manufacturing causes
* **Edge AI** — enables real-time embedded inference

## Industry Impact

For semiconductor manufacturers such as SanDisk, the system enables **earlier defect detection, reduced defect escape, lower scrap, faster process diagnosis, and improved wafer yield** by moving defect intelligence closer to the inspection process.

## Tech Stack

**Python · PyTorch · EfficientNet-B0 · Open-Set Recognition · Autoencoders · Grad-CAM · NumPy · Scikit-learn · Plotly · Google Colab**

