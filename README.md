# PCOS Prediction using Machine Learning

## Overview
Polycystic Ovary Syndrome (PCOS) is a common hormonal disorder among women of reproductive age. This project aims to build a machine learning model to predict the likelihood of a patient having PCOS based on various clinical and physical parameters. 

The project involves comprehensive data preprocessing, feature engineering, handling imbalanced datasets using SMOTE, and applying advanced classification algorithms to achieve reliable predictions.

## Dataset
The dataset contains patient records with multiple clinical features, including:
* **Physical metrics:** Age, Weight, Height, BMI
* **Vital signs:** Pulse rate, Respiratory Rate (RR), Blood Pressure (Systolic & Diastolic)
* **Hormonal & Clinical markers:** Beta-HCG, FSH, LH, AMH, TSH, Prolactin, Vitamin D3, etc.
* **Lifestyle & Symptoms:** Fast food consumption, Regular exercise, Pimples, Hair growth, Skin darkening, Cycle length/regularity.

*(Note: The `Sl. No` and `Patient File No.` columns were dropped as they are irrelevant for modeling).*

## Project Workflow

1. **Data Cleaning & Preprocessing:**
   * Converted string anomalies to proper numeric formats (e.g., handling typos like `'1.99.'` in the `II beta-HCG` column).
   * Handled missing/null values appropriately.
   * Converted necessary columns to `float64` data types.

2. **Feature Engineering:**
   * **BMI Categorization:** Created a new feature categorizing BMI into Underweight, Healthy, and Overweight.
   * **Blood Pressure Categorization:** Grouped Systolic and Diastolic BP into clinical categories (Hypo, Normal, Pre-hyper, Hyper).
   * **LH Levels:** Categorized LH hormone levels into normal and abnormal bins.
   * **One-Hot Encoding:** Applied `pd.get_dummies()` to convert categorical engineered features into numerical formats.

3. **Handling Class Imbalance:**
   * The target variable `PCOS (Y/N)` was highly imbalanced.
   * Used **SMOTE (Synthetic Minority Over-sampling Technique)** from the `imblearn` library to generate synthetic samples for the minority class, ensuring the model trains without bias.

4. **Model Building & Training:**
   * Split the data into Training and Testing sets (80% train, 20% test).
   * Trained the following classifiers:
     * **Random Forest Classifier** (`n_estimators=300`, `max_depth=10`)
     * **XGBoost Classifier** (`learning_rate=0.2`, `max_depth=5`)

5. **Model Evaluation:**
   * Evaluated the model using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
   * Plotted the **ROC & AUC Curve** to visualize the classifier's performance.

## Results
* The model achieved an overall **Accuracy of ~86%**.
* **ROC-AUC Score:** `0.849` (approx. 85%), indicating a strong ability to distinguish between PCOS and non-PCOS patients.

## Technologies Used
* **Python 3**
* **Pandas & NumPy** (Data manipulation)
* **Scikit-Learn** (Machine Learning models, evaluation metrics, data splitting)
* **XGBoost** (Gradient boosting framework)
* **Imbalanced-Learn (SMOTE)** (Handling imbalanced data)
* **Matplotlib & Seaborn** (Data visualization)

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone [(https://github.com/roshni132/PCOS_prediction.git)](https://github.com/roshni132/PCOS_prediction.git)

   cd PCOS-Prediction
   pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
   jupyter notebook CurrentPCOS.ipynb
  
