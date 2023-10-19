Machine Learning II class assignment

**Dataset Description:**
The Breast Cancer Wisconsin dataset contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe various characteristics of cell nuclei present in the image. The target variable is binary, where '0' represents malignant tumors, and '1' represents benign tumors. Here you can find more details: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic


**Exercise:**

**Step 1: Data Loading**
1. Load the Breast Cancer Wisconsin dataset from Scikit-learn using `sklearn.datasets.load_breast_cancer()`.
2. Split the data into features and target variables.

**Step 2: Data Preprocessing**
3. Split the dataset into a training set and a testing set (e.g., 70% train and 30% test).
4. Perform any necessary data preprocessing and feature engineering, such as scaling the features.

**Step 2.5: Feature selection**
5. Apply initial feature selection process (e.g., use some statisical tests like Fisher)

**Step 3** Baseline model
6. Please create simple logistic regression model as a baseline.

**Step 4: AdaBoost Classifier**
7. Train an AdaBoost classifier on the training data.
8. Use cross-validation to find the optimal number of base estimators (n_estimators) for AdaBoost.
9. Tune other hyperparameters (e.g., learning rate) using cross-validation.
10. Visualize the feature importances in the model and try to apply additional feature selection based on it.
11. Evaluate the model's performance on the test set using accuracy, precision-recall curve, and F1-score.

**Step 5: Gradient Boosting Machine (GBM)**
12. Train a Gradient Boosting Machine classifier on the training data.
13. Use cross-validation to find the optimal values for hyperparameters like the number of trees (n_estimators), maximum depth (max_depth), and learning rate.
14. Visualize the feature importances in the model and try to apply additional feature selection based on it.
15. Evaluate the GBM model's performance on the test set using accuracy, precision-recall curve, and F1-score.

**Step 6: Model Comparison and **
16. Compare the performance of the AdaBoost and GBM classifiers and Logistc Regression.
17. Summarize the results and provide insights on which algorithm performed better on this dataset and why.
18. Discuss the impact of hyperparameter tuning on model performance.
