# Small Object Detection in Dense UAV Imagery  
### Structured Ablation of YOLOv8-L on VisDrone2019-DET

This repository presents a structured experimental study on improving small-object detection performance in dense UAV scenes using YOLOv8-L.

The objective is not only to improve mAP, but to analyze *why* architectural and optimization modifications succeed or fail under extreme small-object constraints.

---

## Problem Context

Small-object detection in UAV imagery is challenging due to:

- Severe spatial downsampling in deep backbones  
- Dense object clustering and heavy occlusion  
- Cross-scale assignment conflicts  
- Bounding-box regression instability for tiny objects  

This project investigates these failure modes through controlled architectural ablations.

---

## Experimental Setup

- **Dataset:** VisDrone2019-DET (10 classes, 381k+ annotations)
- **Model:** YOLOv8-L (pretrained)
- **Resolution:** 640 × 640
- **Epochs:** 150
- **Batch Size:** 16
- **Hardware:** NVIDIA A100 (Colab Pro High-RAM)
- **Framework:** PyTorch 2.2+, Ultralytics YOLOv8 (8.3.x)

All training arguments are preserved in the notebooks for full reproducibility.

---

## Structured Ablation Pipeline

### Stage 1 — Baseline YOLOv8-L  
Standard 3-scale detection (P3–P5)

- mAP@0.5: **0.4579**
- mAP@0.5:0.95: **0.2809**

---

### Stage 2 — EMA Attention Integration  
Integrated EMA-style attention inside C2f blocks.

**Goal:** Improve feature selectivity in cluttered UAV scenes.  
**Observation:** Modified confidence calibration but did not surpass baseline globally.

---

### Stage 3 — High-Resolution P2 Head (4-scale P2–P5)  
Extended detector with stride-4 detection head.

**Goal:** Preserve shallow spatial cues for extremely small objects.  
**Result:** Performance degradation due to:
- Candidate explosion  
- Cross-scale gradient conflicts  
- Increased false positives from shallow textures  

**Key Insight:** Resolution expansion alone is insufficient without scale-aware assignment and loss balancing.

---

### Stage 4 — PIoU-Inspired Regression Reweighting  
Implemented regression reweighting strategy within training loop.

**Goal:** Stabilize bounding-box learning for tiny-object localization.  
**Result:** Recovered performance stability relative to naive P2 expansion.

**Conclusion:** Loss shaping was more effective than naive resolution scaling.

---

## SAHI Evaluation (Inference-Time Tiling)

Slice-based inference was evaluated independently.

- Baseline + SAHI improved small-object recall.
- Modified architectures required careful merge/NMS tuning.
- SAHI effectiveness is model-dependent.

---

## Key Technical Findings

1. Small-object regimes are constrained more by regression stability than feature depth.
2. Adding high-resolution heads without assignment tuning can degrade performance.
3. Attention modules alter confidence calibration and require threshold tuning.
4. Inference-time tiling is not universally beneficial.
5. Structured ablation is critical when modifying multi-scale detectors.

---

## Repository Structure
```
small-object-detection-visdrone/
├── README.md
│
├── notebooks/
│   ├── 01_baseline_yolov8l_visdrone.ipynb
│   │      # Baseline YOLOv8-L training on VisDrone (control experiment)
│   └── 02_ablation_sod_yolov8_visdrone.ipynb
│          # Proposed SOD-YOLOv8 modifications + structured ablation study
│
├── configs/
│   └── visdrone.yaml
│          # Dataset configuration file (Kaggle-compatible, path-agnostic)
│
├── results/
│   ├── yolov8_only/
│   │   └── stage_comparison_map.png
│   │          # Baseline vs EMA vs P2 vs PIoU (mAP comparison)
│   │
│   ├── sahi_augmented/
│   │   └── sahi_stage_comparison.png
│   │          # SAHI inference performance comparison
│   │
│   └── calibration_curves/
│       ├── precision_vs_confidence.png
│       ├── recall_vs_confidence.png
│       ├── f1_vs_confidence.png
│       └── precision_vs_recall.png
│          # Calibration and PR analysis curves
│
├── requirements.txt
├── LICENSE
```

## Dataset

The VisDrone dataset is not redistributed.

It can be downloaded programmatically using KaggleHub or directly from the official VisDrone source.

---

## Reproducibility

- Training configuration is preserved in the notebooks.
- All plots and evaluation outputs are included in the `results/` directory.
- Hyperparameters follow Ultralytics defaults unless explicitly modified.
- Hardware used: NVIDIA A100 (40GB).

---

## Author

**Abdul Rafay Mohd**  
Master’s in Artificial Intelligence  

---

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## License

This project is licensed under the terms of the [MIT License](LICENSE).

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## Citation

If you use this repository or report results from it, please cite:
@software{YOLOv8l_Baseline_2025,
  author       = {Abdul Rafay Mohd},
  title        = {YOLOv8l Baseline — VisDrone},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/Mohd-Abdul-Rafay/YOLOv8l}
}

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻

## Contributing

Contributions are welcome. See CONTRIBUTING.md￼.

⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻⸻
