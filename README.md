# Density-Based AQI Analysis

This repository presents a research-oriented analysis of Indian Air Quality Index (AQI) data using density-based clustering techniques. The project focuses on identifying pollution patterns across major Indian cities and understanding their association with air quality severity.

## 📌 Project Overview

Air pollution is a major public health concern in India. Traditional threshold-based analysis often fails to capture complex pollution patterns.  
This project applies **density-based unsupervised learning algorithms** to AQI data to discover hidden structures, outliers, and pollution regimes.

The system also includes an **interactive Streamlit dashboard** for exploratory data analysis and visualization.

## 🎯 Objectives

- Analyze real-world Indian AQI data from multiple cities
- Identify pollution patterns using density-based clustering
- Compare clustering behavior of DBSCAN, HDBSCAN, and OPTICS
- Visualize AQI trends, distributions, and clusters interactively
- Support research and policy-level air quality analysis

## 🧠 Algorithms Used

- **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
- **HDBSCAN (Hierarchical DBSCAN)**
- **OPTICS (Ordering Points To Identify the Clustering Structure)**
- **PCA (Principal Component Analysis)** for dimensionality reduction and visualization

## 📊 Dataset

- **Source:** Indian Government AQI data (city-wise, daily measurements)
- **Features used:**  
  PM2.5, PM10, NO₂, SO₂, O₃, AQI
- **Coverage:** Multiple major Indian cities across several years

> Note: AQI values are capped at 500 for visualization purposes to handle extreme outliers, in accordance with standard AQI interpretation practices.

## 🖥️ Streamlit Dashboard Features

- City-wise AQI selection
- AQI distribution plots per city
- Top 10 most polluted cities (average AQI)
- Density-based clustering analysis
- PCA-based 2D visualization of clusters
- Cluster-wise pollution profile analysis

## Project Structure

Density-Based-AQI-Analysis/
├── app.py                  # Streamlit dashboard application
├── AQI_Clustering.ipynb    # Jupyter Notebook (analysis and experiments)
├── data/
│   └── city_day.csv        # AQI dataset
├── .gitignore
└── README.md

## How to Run the Project

Follow the steps below to set up and run the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/Amit12hangargi/Density-based-AQI-Analysis.git

cd Density-based-AQI-Analysis

pip install streamlit pandas numpy matplotlib seaborn scikit-learn hdbscan

streamlit run app.py
