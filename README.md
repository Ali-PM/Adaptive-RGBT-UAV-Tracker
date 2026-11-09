```markdown
# Adaptive RGB-T UAV Tracker with Zero-Drift Kalman Filter

An advanced, dual-modality (Visible RGB + Thermal Infrared) UAV tracking pipeline built on top of the **SiamFC** architecture. This project introduces a dynamic modality quality-confidence fusion mechanism alongside a trajectory-decoupled Kalman Filter to robustly track Unmanned Aerial Vehicles (UAVs) under complex environments, adverse lighting, and severe occlusions.

---

## 🚀 Key Features

* **Adaptive Dual-Modality Fusion:** Dynamically measures the visual quality of RGB and Thermal patches (Laplacian variance and contrast) to adjust real-time modality fusion weights ($\alpha_t$).
* **Dual-Flow Semantic Modulator:** Leverages cross-attention response mappings to preserve target representation during sensor degradation.
* **Zero-Drift & Outlier-Rejection Kalman Filter:**
  * Filters high-frequency sub-pixel jitter during stable tracking to optimize Center Location Error (CLE) and bounding box overlap.
  * Restrains bounding box jumping (Outlier Rejection) during sudden target occlusions or clutter.
  * Decouples the tracking search window from the output bounding box to eliminate recursive drift accumulation.
* **Automated Evaluation Pipeline:** End-to-end framework computing Overall Precision (OPE), Success Rate (mSA), and Center Location Error (CLE) across the **Anti-UAV** benchmark, automatically generating summary plots and telemetry logs.

---

## 📊 Benchmark & Ablation Study

Evaluated on the **Anti-UAV** dataset across diverse sequence challenges:

| Method / Configuration | Precision (%) | Success Rate (%) | Mean CLE (px) | FPS |
| :--- | :---: | :---: | :---: | :---: |
| 1. RGB Only | 38.17 | 20.23 | 94.0 | 194.2 |
| 2. Thermal Only | 64.33 | 61.69 | 85.1 | 192.4 |
| 3. Fixed RGB-T Fusion | 62.96 | 60.90 | 91.7 | 170.8 |
| 4. Adaptive (Quality Only) | 65.97 | 61.07 | 82.3 | 168.5 |
| 5. Adaptive (Quality + Confidence) | 72.61 | 69.55 | 57.4 | 162.8 |
| 6. Adaptive + Temporal Smoothing | 74.54 | 71.37 | 54.6 | 160.4 |
| **7. Proposed Full (+ Kalman Filter)** | **74.65** | **71.61** | **53.4** | **159.3** |

---

## 🛠️ Tech Stack & Dependencies

* **Deep Learning Framework:** PyTorch
* **Computer Vision:** OpenCV, NumPy
* **Data Processing & Visualization:** Pandas, Matplotlib, OpenPyXL
* **Automation:** Python `requests`, `python-docx`

---

## 📁 Repository Structure

```text
├── adaptive_fusion_module.py   # Quality assessor, semantic modulator & Kalman filter
├── siamfc_adaptive_rgbt.py     # Siamese tracking architecture & runtime tracker
├── run_ablation_experiments.py # Sequence evaluation, IoU & CLE metrics calculation
├── run_evaluation_pipeline.py  # End-to-end multi-mode evaluation & plot generator
├── results/                    # Generated ablation comparison and OPE curves
└── README.md

```

---

## ⚙️ Quick Start

### 1. Clone the Repository

```bash
git clone [https://github.com/Ali-PM/Adaptive-RGBT-UAV-Tracker.git](https://github.com/Ali-PM/Adaptive-RGBT-UAV-Tracker.git)
cd Adaptive-RGBT-UAV-Tracker

```

### 2. Install Requirements

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib requests openpyxl python-docx

```

### 3. Run Benchmark Evaluation

Place the pretrained model weights (`model.pth`) and validation sequences inside `dataset_root/val/`:

```bash
python run_evaluation_pipeline.py

```

---

## 👨‍💻 Author

**Ali Molaei**

M.Sc. in Mechatronics Engineering | Computer Vision & Backend Developer

* Email: [molaeiali24@gmail.com](https://www.google.com/search?q=mailto%3Amolaeiali24%40gmail.com)
* Telegram: [@AliMoj4z](https://www.google.com/search?q=https://t.me/AliMoj4z)
