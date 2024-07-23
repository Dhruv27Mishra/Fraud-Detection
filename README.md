# Fraud-Detection
Objective of the Fraud Detection Program


Overview

The objective of this program is to develop and evaluate a machine learning model for detecting fraudulent transactions in a financial dataset. Fraud detection is a critical task in the financial industry, aimed at identifying suspicious transactions that may indicate fraudulent activity. This program involves several steps, including data preprocessing, model training, evaluation, and visualization of the results.


Model Used: XGBoost Classifier

The model used in this fraud detection program is the XGBoost Classifier, a highly efficient and powerful implementation of gradient boosting. XGBoost stands for eXtreme Gradient Boosting, and it is widely used in machine learning competitions and real-world applications due to its superior performance and scalability.


Detailed Objectives

Data Preprocessing:

Load Data: Import transaction data from CSV files located in a specific directory.
Feature Extraction: Convert the transaction date to individual components like year, month, day, and hour.
Encoding: Transform categorical features (e.g., transaction type, customer ID, merchant ID) into numerical values suitable for machine learning algorithms.

Model Training:

Model Selection: Use the XGBoost classifier, a powerful gradient boosting algorithm, for training the fraud detection model.
Training: Fit the model on the training dataset to learn patterns and relationships between features and the target variable (fraudulent or non-fraudulent transaction).

Model Evaluation:

Predictions: Make predictions on the test dataset to assess the model's performance.
Metrics Calculation: Calculate various evaluation metrics, including accuracy, confusion matrix, ROC curve, and Precision-Recall curve.

Visualization:

Confusion Matrix: Visualize the number of true positives, false positives, true negatives, and false negatives to understand the model's classification performance.
ROC Curve: Plot the Receiver Operating Characteristic (ROC) curve to visualize the trade-off between the true positive rate and false positive rate.
Precision-Recall Curve: Plot the Precision-Recall curve to evaluate the balance between precision (positive predictive value) and recall (sensitivity).

Model Persistence:

Save Model: Save the trained model to a file for future use, allowing it to be loaded and applied to new data without retraining.


How to run the code:
To use the program follow the given steps below:
1.) Go to Directory "Dataset" 
2.) Run the program gen.py which generates two files train_transactions.csv and test_transactions_csv 
3.) Run the program tester.py to use the prediction ML model and get results in the terminal 
4.) Run graph.py in the main directory to get grphical reults for the same 
