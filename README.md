# Fraud Detection Analytics  

## Project Overview  
This project focuses on detecting fraudulent claims using machine learning. We analyze patterns in the dataset, apply feature engineering, and train multiple models to improve fraud detection accuracy. The primary goal is to minimize financial losses by identifying fraudulent claims effectively while balancing false positives and false negatives.  

## Dataset Description  
The dataset used in this project is [`financial_data_log.csv`](financial_data_log.csv). It contains transaction details, customer demographics, and fraud indicators. Since fraudulent claims are much fewer than normal ones, the dataset is highly imbalanced, requiring special techniques during preprocessing and model training.  

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
5. **[Clustering for Fraud Detection](#clustering-for-fraud-detection)** – Using **K-Means clustering** to detect fraudulent transactions and visualizing results with **PCA**.  

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

### **5. Clustering for Fraud Detection**  
Fraudulent transactions often follow different patterns than normal ones. We used **K-Means clustering** to group similar transactions and detect anomalies.  

#### **Steps Taken:**  
1. **Data Scaling** to normalize features for better clustering.  
2. **Clustering** using **K-Means (K=2)** to group transactions into **normal** and **fraudulent** clusters.  
3. **Used PCA** (Principal Component Analysis) to **reduce dimensions** and visualize transaction clusters. **Insights from visualization:**  
   - Since the **non-fraud transactions (blue) are dense at the bottom left**, it means most normal transactions have **smaller amounts** and follow a **similar pattern**, so
     they are grouped closely together.
   - **Fraud transactions (red) are spread out**, meaning they have **different patterns and larger amounts**.
   - Some fraud points are **far from the main cluster**, showing they behave very differently.

Detailed code and analysis are available in [`fraud_analytics.ipynb`](fraud_analytics.ipynb).
