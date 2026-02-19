# YOLOv8l Baseline — VisDrone

This repository provides a **reproducible baseline** for training the YOLOv8l model on the **VisDrone2019-DET** dataset.  
All code, paths, weights, and results are preserved exactly as generated.  
The dataset is **not redistributed**; instead, you can fetch it automatically via **KaggleHub**.  

> **Environment:** All experiments were run on **Google Colab Pro** with **High-RAM runtime** and an **NVIDIA A100 GPU**.

---

## 📂 Repository Structure
```
small-object-detection-visdrone/
├── README.md
├── notebooks/
│   ├── 01_baseline_yolov8l_visdrone.ipynb
│   └── 02_ablation_sod_yolov8_visdrone.ipynb
├── configs/
│   └── visdrone.yaml
├── results/
│   ├── yolov8_only/
│   │   └── stage_comparison_map.png
│   ├── sahi_augmented/
│   │   └── sahi_stage_comparison.png
│   └── calibration_curves/
│       ├── precision_vs_confidence.png
│       ├── recall_vs_confidence.png
│       ├── f1_vs_confidence.png
│       └── precision_vs_recall.png
├── requirements.txt
├── LICENSE
└── .gitignore

## 📊 Dataset

We do not redistribute the dataset. Download it programmatically with KaggleHub:

```bash
pip install kagglehub
```

```python
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("banuprasadb/visdrone-dataset")
print("Path to dataset files:", path)


DATA_ROOT = Path(path) / "VisDrone"
print("DATA_ROOT:", DATA_ROOT)
```

On Kaggle, the dataset is available at:
/kaggle/input/visdrone-dataset/VisDrone_Dataset/
├─ VisDrone2019-DET-train/
├─ VisDrone2019-DET-val/
├─ VisDrone2019-DET-test-dev/
├─ VisDrone2019-DET-test-challenge/
└─ visdrone.yaml

```yaml
path: /kaggle/input/visdrone-dataset/VisDrone_Dataset
train: VisDrone2019-DET-train/images
val:   VisDrone2019-DET-val/images
test:  VisDrone2019-DET-test-dev/images
names: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## ⚙️ Training Configuration

All hyperparameters are frozen in `args.yaml`.  
Key parameters:

- **Model**: `yolov8l.pt`  
- **Epochs**: 150  
- **Image size**: 640 × 640  
- **Batch size**: 16  
- **Workers**: 8  
- **Device**: GPU (A100, Colab Pro High-RAM)  
- **Framework**: Ultralytics YOLOv8 v8.3.5  
- **Torch**: 2.2+  


⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 📈 Results

All key figures used for reporting are stored in `results/`:

- `results/yolov8_only/stage_comparison_map.png`: mAP comparison across stages (YOLO-only)
- `results/sahi_augmented/sahi_stage_comparison.png`: SAHI mAP@0.5 comparison across stages
- `results/calibration_curves/`:
  - `precision_vs_confidence.png`
  - `recall_vs_confidence.png`
  - `f1_vs_confidence.png`
  - `precision_vs_recall.png`

Raw training artifacts (e.g., `results.csv`, `args.yaml`, `best.pt`) are intentionally **not tracked** in GitHub to keep the repository lightweight and reproducible. You can regenerate them by running the notebooks in `notebooks/`.


⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 🚀 Usage

### Clone and install
```bash
git clone https://github.com/Mohd-Abdul-Rafay/small-object-detection-visdrone.git
cd small-object-detection-visdrone
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

Open YOLOv8l Baseline.ipynb in Colab Pro (High-RAM, A100 GPU) or Jupyter.
Mount the dataset (via KaggleHub or manually) and run the cells.

## Inference
```bash
from ultralytics import YOLO
model = YOLO("runs/yolov8_training/train/weights/best.pt")
model.predict(source="path/to/images", imgsz=640, save=True)
```

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 🧩 Reproducibility

- Notebook and outputs are preserved exactly.  
- Dataset is external via KaggleHub.  
- Weights and exports tracked with Git LFS.  
- CI workflow (`.github/workflows/smoke.yml`) validates environment and imports.  
- Training confirmed on **Colab Pro High-RAM A100 GPU runtime**.  

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 📜 License

This project is licensed under the terms of the [MIT License](LICENSE).

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 📚 Citation

If you use this repository or report results from it, please cite:
@software{YOLOv8l_Baseline_2025,
  author       = {Abdul Rafay Mohd},
  title        = {YOLOv8l Baseline — VisDrone},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/Mohd-Abdul-Rafay/YOLOv8l}
}

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 🤝 Contributing

Contributions are welcome. See CONTRIBUTING.md￼.

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## 🔒 Security

See SECURITY.md￼ for vulnerability reporting.
---
```bash
![build](https://github.com/<your-username>/YOLOv8l/actions/workflows/smoke.yml/badge.svg)
```
