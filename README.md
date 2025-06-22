# Leak Detection Classifier

A classification model for detecting types of water pipeline leaks based on sensor data, using a hybrid DTKNN (Decision Tree + K-Nearest Neighbors) algorithm.

> 🚀 **Note:** Our team developed a custom hybrid model where important features are selected via a Decision Tree and classification is performed using a distance-based KNN. All performance evaluations, including confusion matrices and metrics, were implemented manually.

---

## 📌 Project Overview

This project was completed in Spring 2025 as part of a classification modeling and evaluation assignment. The objective was to develop an effective model to classify five leak-related categories using real-world sensor data.

- **Goal**: Predict leak type from water pipeline sensor data
  
- **Classes**:
  - `Out`: Outdoor leakage  
  - `In`: Indoor leakage  
  - `Noise`: Electrical/mechanical noise  
  - `Others`: Environmental noise  
  - `Normal`: Normal/no leak

---

## 📁 Dataset

- **Source**: [AIHub 누수 탐지 데이터셋 (2019~2022)](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=138)
  
- **Files**:
  - `Classification_training_data.csv`
  - `Classification_testing_data.csv`
    
- **Features**:
  - `lrate`, `llevel`, frequencies from `0Hz` to `5120Hz`, and `MAX0` to `MAX19`

---

## 🧠 Model: DTKNN (Hybrid Approach)

1. **Feature Selection**:  
   - Decision Tree used to rank and select the most informative features (e.g., `lrate`, `370Hz`, `330Hz`, ...)
   - Reduced dimensions improve both performance and speed

2. **Classification**:
   - KNN applied using Euclidean distance
   - Best performance achieved at **K=1**

3. **Evaluation**:
   - Confusion matrix calculated manually
   - Metrics reported: Accuracy, Precision, Recall

---

## 📊 Results

| Metric     | Value |
|------------|--------|
| Accuracy   | **0.93** |
| Precision  | **0.93** |
| Recall     | **0.93** |

- K=1 was optimal due to strong local clustering of data
- Distance-based classification performed best with DT-selected features
- Evaluation fully implemented without sklearn metrics

---

## 🚀 How to Run

```bash
python src/test.py
python src/main.py

