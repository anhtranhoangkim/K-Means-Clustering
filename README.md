# 🌍 K-Means Clustering for Country Classification

This project applies the **K-Means clustering algorithm** to group countries based on key socio-economic and health indicators. The goal is to help humanitarian organizations strategically allocate financial aid to countries most in need during crises and disasters.

Developed for the Artificial Intelligence course at National Economics University, this project focuses on real-world data application, dimensionality reduction, clustering, and visualization.

---

## 🔗 Run on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OT6Z_VVYzy4ms3PUwOE5Q8t7HL8R-27Y?usp=sharing)

---

## 🎯 Project Objectives

- Group 167 countries based on 9 quantitative attributes (e.g., GDP, life expectancy, child mortality).
- Identify clusters of countries that are **most in need of aid**.
- Provide an **interactive map** to visualize the clustering result.

---

## 📂 Dataset

- **File**: `Country-data.csv` (included in the repository)
- **Rows**: 167 countries
- **Attributes**:
  - `child_mort`: Child mortality rate
  - `exports`, `imports`, `health`: Economic metrics (% of GDP)
  - `income`, `inflation`, `life_expec`, `total_fer`, `gdpp`: Key socio-economic indicators

---

## ⚙️ Features

- 📊 **Data Preprocessing**: Handle missing values, normalization, and dimensionality reduction
- 🔁 **Clustering**: K-Means algorithm implementation using Scikit-learn
- 📈 **Evaluation**: Elbow method to find optimal `k`
- 🌐 **Visualization**:
  - Static plots via Matplotlib & Seaborn
  - Interactive world map with Plotly showing cluster assignments
- 🖼️ **Color legend** for aid need:
  - 🔴 Red – High need
  - 🟡 Yellow – Medium need
  - 🟢 Green – No urgent need

---

## 🧪 Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Plotly, Kaleido
- Google Colab

---

## 🚀 How to Run

1. Open the [Colab notebook](https://colab.research.google.com/drive/1OT6Z_VVYzy4ms3PUwOE5Q8t7HL8R-27Y?usp=sharing)
2. Upload `Country-data.csv`
3. Run the cells sequentially
4. View:
   - Data preprocessing and dimensionality reduction
   - K-means clustering results
   - Interactive world map

---

## 👩‍💻 Author
**Trần Hoàng Kim Anh**

Artificial Intelligence Course Project – 2023

National Economics University

---

## 📄 Documentation & Report

(in Vietnamese)

📄 [View Full Report (PDF)](https://github.com/anhtranhoangkim/K-Means-Clustering/blob/main/docs/K-Means%20Clustering%20-%20Tran%20Hoang%20Kim%20Anh.pdf)
