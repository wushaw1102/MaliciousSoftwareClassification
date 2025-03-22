
---

# Malware Classification Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg) ![LightGBM](https://img.shields.io/badge/LightGBM-3.x-green.svg) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A multi-class classification project using LightGBM to identify 9 types of malware. Built with Python, optimized with 5-fold cross-validation, and includes visualizations like confusion matrix and ROC curve.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dataset](#dataset)
6. [Results](#results)
7. [Visualizations](#visualizations)
8. [License](#license)

---

## Project Overview
This project aims to classify malware into 9 distinct categories using a LightGBM-based multi-class classification model. It leverages Python libraries like Pandas, Scikit-learn, and LightGBM, with 5-fold stratified cross-validation for robust performance evaluation.

---

## Features
- Multi-class classification (9 classes) using LightGBM.
- 5-fold cross-validation for model optimization.
- Evaluation metrics: accuracy, precision, recall, F1-score, and AUC.
- Visualizations: confusion matrix, ROC curve, feature importance, and learning curve.
- Feature importance analysis based on gain and split metrics.

---

## Installation

### Prerequisites
- Python 3.8
- Required libraries:
  ```bash
  pip install numpy pandas scikit-learn lightgbm matplotlib seaborn
  ```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/ushaw1102/MaliciousSoftwareClassification.git
   cd MaliciousSoftwareClassification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Ensure the dataset files (`train.csv` and `test.csv`) are placed in the `./datasets/` directory.
2. Run the main script:
   ```bash
   python main.py
   ```
3. Outputs:
   - Predictions saved as `submit.csv`.
   - Evaluation metrics saved as `evaluation.csv`.
   - Visualizations saved in `./results/` folder.

---

## Dataset
- **Source**: Custom dataset (not included due to size/privacy).
- **Structure**:
  - `train.csv`: Features and `label` column (9 classes).
  - `test.csv`: Features and `id` column for prediction.
- Replace with your own dataset if needed, ensuring column names match the code.

---

## Results
- **Accuracy**: [0.92]
- **F1-Score**: [0.90]
- **AUC**: [0.95]
- Full evaluation metrics are available in `evaluation.csv`.

---

## Visualizations
Visual outputs are saved in the `./results/` directory:
- **Confusion Matrix**: `hunjiaojuzhen.png`
- **ROC Curve**: `roc.png`
- **Learning Curve**: `learning_curve.png`
- **Feature Importance**: (Uncomment code to generate `tezheng.png`)

Example:
![Confusion Matrix](./results/hunjiaojuzhen.png)

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.


