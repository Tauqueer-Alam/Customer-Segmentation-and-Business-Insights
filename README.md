# Mall Customer Segmentation Dashboard

An interactive, premium analytics dashboard built with **Streamlit** and **Plotly** to explore, profile, and target customer segments. The project showcases the application of unsupervised machine learning (K-Means, DBSCAN, Hierarchical Clustering) integrated with a supervised proxy classifier to enable out-of-sample customer predictions in real-time.

---

## 🚀 Key Features

* **Multi-Algorithm Control Panel**: Swap between **K-Means**, **DBSCAN**, and **Hierarchical Clustering** with on-the-fly hyperparameter tuning (number of clusters $K$, linkage criteria, Epsilon radius, and minimum samples).
* **Supervised KNN Predictor Wrapper**: Implements a 3-Nearest Neighbors classifier trained dynamically on active cluster labels to enable real-time predictions for models that do not natively support out-of-sample data points (DBSCAN & Hierarchical).
* **Dynamic Rule-Based Segment Profiling**: Centroids and demographics are processed dynamically to label segments (e.g., *Affluent Spenders*, *Frugal Shoppers*) and generate tailored business marketing strategies. DBSCAN outliers (Cluster `-1`) are captured and handled automatically.
* **Interactive EDA Visualizations**:
  * 2D & 3D Interactive Scatter Plots with custom sizing/colors representing customer characteristics.
  * Demographic histograms/distributions.
  * Segment representation breakdown pie/bar charts.
* **Single Predictor Tool**: Enter demographics (Gender, Age, Income, Spending Score) to categorize a new customer and compare their profile vs. the segment average.
* **Batch Segmentation Tool**: Upload a CSV of customer records, perform batch inference, view cohort representations, and download the results as a new CSV.

---

## 🛠️ Technology Stack

* **Front-End & Dashboard**: Streamlit
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-Learn (`KMeans`, `DBSCAN`, `AgglomerativeClustering`, `KNeighborsClassifier`, `StandardScaler`)
* **Data Visualization**: Plotly (Express & Graph Objects)
* **Pipeline Preservation**: Pickle

---

## 📂 Project Structure

```
├── Mall_Customers.csv      # Raw customer dataset
├── app.py                  # Core Streamlit dashboard and UI logic
├── train_model.py          # Machine learning pipeline generation (Scaler & initial K-Means)
├── scaler.pkl              # Saved StandardScaler instance
├── kmeans_model.pkl        # Pre-trained K-Means model artifact
├── model.ipynb             # Jupyter Notebook containing early EDA and modeling experiments
└── README.md               # Project documentation
```

---

## ⚙️ Setup and Installation

### 1. Clone the Workspace / Repository
Download or navigate to the project directory:
```bash
cd "Project - Customer Segmentation"
```

### 2. Install Dependencies
Install the required packages using pip:
```bash
pip install streamlit pandas numpy scikit-learn plotly
```

### 3. Initialize ML Pipeline Artifacts (Optional)
Generate the serialized scaler and K-Means models:
```bash
python train_model.py
```

### 4. Run the Streamlit Application
Start the dashboard locally:
```bash
streamlit run app.py
```
The app will automatically open in your default browser at `http://localhost:8501`.

---

## 📊 Dashboard Workspaces

### 1. 📈 Dashboard & Insights
* Displays overall database metrics (Total Customers, Averages).
* Dynamically lists cards for each discovered segment, including size, average demographics, and a data-driven business strategy.
* DBSCAN outlier points are grouped as a specific noise segment with security/bespoke recommendations.

### 2. 📊 Exploratory Data Analysis
* **Scatter Visualizer**: Compare two features in 2D or 3D, using bubble sizes to represent age.
* **Variable Distributions**: Inspect demographic distributions separated by cluster.
* **Segment Breakdown**: View pie charts displaying cohort market share.

### 3. 🎯 Single Predictor
* Input age, gender, income, and spending score.
* The system projects the customer onto the scaled feature space and classifies them using the dynamic KNN classifier wrapper.
* Compares the input metrics against the segment averages using custom interactive gauge charts.

### 4. 📁 Batch Segmentation
* Upload customer lists in CSV format (requiring `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)` columns).
* Batch infer their segments, inspect cohort distributions, and export/download the segmented spreadsheet.
