# Fraud Detection Analytics  

## Project Overview  
This project focuses on detecting fraudulent claims using machine learning. We analyze patterns in the dataset, apply feature engineering, and train multiple models to improve fraud detection accuracy. The primary goal is to minimize financial losses by identifying fraudulent claims effectively while balancing false positives and false negatives.  

## Dataset Description  
The dataset used in this project is [`financial_data_log.csv`](financial_data_log.csv). It contains records of mobile money transactions, including transaction details, customer demographics, and fraud indicators. Fraudulent transactions are much rarer than normal ones, resulting in a highly imbalanced dataset.

Below is the description of the dataset columns:  

| Column           | Description |
|-----------------|-------------|
| `step`          | Represents a unit of time in the real world, with 1 step equating to 1 hour. The total simulation spans 744 steps, equivalent to 30 days. |
| `type`          | Transaction types include CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER. |
| `amount`        | The transaction amount in the local currency. |
| `nameOrig`      | The customer initiating the transaction. |
| `oldbalanceOrg` | The initial balance before the transaction. |
| `newbalanceOrig`| The new balance after the transaction. |
| `nameDest`      | The transaction's recipient customer. |
| `oldbalanceDest`| The initial recipient's balance before the transaction. Not applicable for customers identified by 'M' (Merchants). |
| `newbalanceDest`| The new recipient's balance after the transaction. Not applicable for 'M' (Merchants). |
| `isFraud`       | Identifies transactions conducted by fraudulent agents aiming to deplete customer accounts through transfers and cash-outs. |
| `isFlaggedFraud`| Flags large-scale, unauthorized transfers between accounts, with any single transaction exceeding 200,000 being considered illegal. |

## Task Description  
1. **[Data Preprocessing](#data-preprocessing)** – Cleaning and preparing the dataset, ensures we have a well-structured dataset for modeling.  
2. **[Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)** – Understanding the characteristics of fraudulent vs. normal claims using visualizations and statistics.  
3. **[Weight of Evidence (WOE) for Fraud Detection](#weight-of-evidence-woe-for-fraud-detection)** – Transforming categorical features into numerical values based on fraud likelihood.
4. **[Fraud Detection Using Outlier Detection](#fraud-detection-using-outlier-detection)** – Using statistical methods (IQR and Z-score) to detect fraudulent transactions.
5. **[Clustering for Fraud Detection](#clustering-for-fraud-detection)** – Using K-Means clustering to detect fraudulent transactions and visualizing results with **PCA**.
6. **[Model Evaluation](#model-evaluation)** – Comparing statistical vs. clustering models using performance metrics
7. **[Ensemble Models for Fraud Detection](#ensemble-models-for-fraud-detection)** – Combining multiple models to improve fraud detection.
8. **[Financial Evaluation](#financial-evaluation)** – Analyzing the cost-benefit impact of fraud detection models.

---

## Data Preprocessing
To ensure data quality and prevent errors in model training, we performed the following steps:  

1. **Loaded the dataset** (`financial_data_log.csv`) using Pandas.  
2. **Checked for missing values** and confirmed no significant gaps in the dataset.  
3. **Checked for duplicate records** and found that `<3>` duplicate rows were `<removed/none found>`.  
4. **Separated features (`X`) and target (`y`)**, where `isFraud` is the fraud label (0 = not fraud, 1 = fraud).  
5. **Split the data into training (80%) and testing (20%) sets** while maintaining the fraud ratio.

This step ensures the dataset is clean and ready for feature engineering and modeling.  
The detailed code and results can be found in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).  

## Exploratory Data Analysis (EDA)
Understanding the dataset is crucial before applying any model. We performed **EDA** to detect **patterns, relationships, and outliers** that could indicate fraud.

1. **Univariate & Bivariate Analysis**
   - **8 histograms** to check the distribution of features.
   - **7 boxplots** comparing key features with fraud status.
2. **Multivariate Analysis**
   - **2 scatter plots** to visualize relationships between balances, amount, and fraud.
   - **Correlation matrix** to find highly related features.  
3. **Outlier Detection** : Identified extreme transaction amounts that could be linked to fraud.  
4. **Key Insights**
   - Fraudulent transactions tend to involve **very high amounts**, as shown by the right-skewed distribution of amount, while most transactions remain small.
   - Fraud cases often **start with higher balances**, and destination balances show a wide range.
   - There is a **strong positive correlation** between `oldbalanceOrg` and `newbalanceOrig`* and between `oldbalanceDest` and `newbalanceDest`, suggesting that many transactions
     maintain balance consistency.
   - High-value transactions are not always fraudulent, but most fraud cases involve large amounts, making **extreme transaction values an important red flag** for detection.

These insights help in feature selection for fraud detection modeling.  
The detailed analysis can be found in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).  

## **Weight of Evidence (WOE) for Fraud Detection**  
To improve fraud prediction, we used **Weight of Evidence (WOE)** to transform categorical features into numerical values based on their relationship with fraud. The process involved:  

1. **Binning numerical variables** to create meaningful categories.  
2. **Computing WOE values** for each category to measure its association with fraud.  
3. **Calculating Information Value (IV)** to assess feature importance.  
4. **Key Findings:**  
   - `amount` and `type` both show **Strong IV**, meaning they are **highly predictive of fraud**.  
   - Fraud is **more likely in transactions with higher amounts and the TRANSFER type**.  

The detailed WOE analysis can be found in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).  

## **Fraud Detection Using Outlier Detection**
To identify fraud, we applied **Interquartile Range (IQR) and Z-score** to detect extreme outliers. We performed the following steps:

1. **Created functions for IQR and Z-score detection.**
2. **Selected 2 variables from EDA** (`newbalanceOrig`, `oldbalanceOrg`) and **2 from WOE** (`amount`, `type_WOE`).
3. **Applied IQR method** to detect outliers in the selected variables.
4. **Applied Z-score method** to detect outliers in 3 numerical variables (`amount`, `oldbalanceOrg`, `newbalanceOrig`).
5. **Summarized results using a proportion score and majority vote** to classify fraudulent transactions, resulting these findings:
   - The **Precision** score shows that the model is good at predicting non-fraud cases correctly, but it often misclassifies fraud cases (many false positives).
   - The **Recall** score indicates that out of all actual fraud cases, only **6%** were identified as fraud.
   - The model might be biased towards non-fraud cases due to **class imbalance**.

The detailed analysis can be found in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).

## **Clustering for Fraud Detection**  
Fraudulent transactions often follow different patterns than normal ones. We used **K-Means clustering** to group similar transactions and detect anomalies. Steps Taken:  
1. **Data Scaling** to normalize features for better clustering.  
2. **Clustering** using **K-Means (K=2)** to group transactions into **normal** and **fraudulent** clusters.  
3. **Used PCA** (Principal Component Analysis) to **reduce dimensions** and visualize transaction clusters. **Insights from visualization:**  
   - Since the **non-fraud transactions (blue) are dense at the bottom left**, it means most normal transactions have **smaller amounts** and follow a **similar pattern**, so
     they are grouped closely together.
   - **Fraud transactions (red) are spread out**, meaning they have **different patterns and larger amounts**.
   - Some fraud points are **far from the main cluster**, showing they behave very differently.

Detailed code and analysis are available in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).

## **Model Evaluation**  
After building fraud detection models, we evaluated their performance using three key metrics:  

- **Precision**: How many predicted fraud cases were actually fraud?  
- **Recall**: How many actual fraud cases did the model successfully detect?  
- **F1-score**: The balance between precision and recall.  

The evaluation results, detailed in [`fraud_analytics.ipynb`](fraud_analytics.ipynb), show that **neither model performs well for fraud detection** because they do not learn from labeled fraud cases. A **supervised model like Logistic Regression or Random Forest**, trained on actual fraud labels, would be more effective.  

### **Comparison of Models**  

| Model | Strengths | Weaknesses |  
|--------|------------|-------------|  
| **Statistical Model (Outlier Detection)** | Simple and easy to understand. Does not require labeled data. | Misses most fraud cases (low recall). Many frauds do not appear as outliers. |  
| **Clustering (K-Means)** | Can find hidden patterns in data. Works without labeled fraud cases. | Clusters are not meaningful for fraud detection. Too many false positives. |  

## **Ensemble Models for Fraud Detection**  
To improve fraud detection, we tested three models that combine **statistical methods** and **clustering**:  

1. **Model 1 (IQR Method)** – Identifies fraud based on outliers in the interquartile range.  
2. **Model 2 (Z-Score Method)** – Flags fraud using standard deviation thresholds.  
3. **Model 3 (IQR + Z-Score + K-Means)** – Combines statistical outlier detection with clustering.  

The results are detailed in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).
### **Evaluation Metrics**  

| Model | Precision | Recall | F1-score |  
|-------|-----------|--------|----------|  
| **Model 1 (IQR)** | 0.0141 | 0.0277 | 0.0187 |  
| **Model 2 (Z-Score)** | 0.0374 | 0.0116 | 0.0177 |  
| **Model 3 (IQR + Z-Score + K-Means)** | 0.0137 | 0.0214 | 0.0168 |  

Model 3 **usually performs better**, but **only if the individual models work well**.

## **Financial Evaluation**  
To assess the financial impact of fraud detection, we evaluated the **best-performing model** based on **F1-score** and calculated its total utility, cost, and return on investment (ROI). The following steps were performed:  

1. **Selected the best model** based on the highest **F1-score**.  
2. **Identified True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN)** to categorize model performance.  
3. **Calculated financial metrics**:
   - **Prevented Fraud Loss**: The total amount saved by correctly detecting fraudulent transactions (**TP**).  
   - **Loss from Missed Fraud**: The total amount lost due to undetected fraud cases (**FN**).  
   - **False Positive Cost**: The financial loss from incorrectly flagging legitimate transactions as fraud (**FP**).  
   - **Profit from Legitimate Transactions**: The gain from correctly identifying non-fraudulent transactions (**TN**).  
4. **Computed the Return on Investment (ROI)** to assess if the fraud detection system is financially viable.  

### **Findings**  
- **Best Model**: Model 1 (IQR)  
- **Total Utility (Net Profit)**: **-545,114,696.98** → The model results in a **financial loss**.  
- **Total Cost of Ownership**: **800,000,000** → The operational cost is high.  
- **Total Cost of Fraud Handling**: **228,600,000** → Cost incurred to manage fraud cases.  
- **ROI**: **-53.00%** → A negative ROI indicates that the fraud detection system is **not profitable** and incurs more losses than savings.  

The results suggest that the current fraud detection model needs improvement, as the **cost of false positives and missed fraud cases outweighs the benefits** of detecting fraud.
