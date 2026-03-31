# SLIC Superpixel & Grad-CAM Project

## Overview

This project contains two main components:

1. **Simple Linear Iterative Clustering (SLIC) Superpixels**
   A custom implementation of the SLIC algorithm for generating superpixels from images.

2. **Visual Attention in Deep Neural Networks (Grad-CAM)**
   A visualization technique to highlight important regions in images used by deep learning models for prediction.

---

## How to Run

### Install Dependencies 
```bash
pip install -r requirements.txt
```

### Task 1 — SLIC Superpixel Segmentation

Run from the **project root directory**:

```bash
python main.py
```

**Input:**

* Place images inside: `images/input/`

**Output:**

* Results saved to: `images/output/`

---

### Task 2 — Grad-CAM (Visual Attention)

Run from the **project root directory**:

```bash
python task2/gradcam_experiment.py
```
This will:

* Load an image
* Run a CNN model
* Generate Grad-CAM heatmaps
* Save or display attention visualizations

---

## Project Structure

```
SLIC-SUPERPIXEL/
│
├── images/
│   ├── input/                  # Input images
│   └── output/                 # Output results
│
├── task2/
│   └── gradcam_experiment.py   # Task 2: Grad-CAM visualization
│
├── slic_custom.py              # Core SLIC algorithm implementation
├── utils.py                    # Helper functions 
├── main.py                     # Task 1: Entry point for SLIC
│
├── requirements.txt            # All dependencies
├── README.md                   
└── .gitignore
```

---
