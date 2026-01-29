# adult-income-classification-random-forest
A machine learning analysis using the Adult dataset to classify income levels. The workflow includes data cleaning, exploratory data analysis, feature encoding, Random Forest classification, class imbalance handling using SMOTE, hyperparameter tuning, and performance evaluation.
This repository contains a machine learning workflow developed using the Adult Income dataset to classify individuals into income groups based on demographic and employment-related features.

The analysis begins with data loading and preprocessing, where missing values are handled and irrelevant records are removed. The target variable is converted into a binary format representing income levels. Categorical attributes such as workclass, education, occupation, marital status, and native country are transformed into numerical values using label encoding to prepare the data for model training.

Exploratory data analysis is performed to understand patterns within the dataset. Visualizations are used to examine income distribution across education levels and age groups, highlighting differences between low- and high-income categories.

A Random Forest classifier is trained using a stratified train-test split to maintain class balance during evaluation. Model performance is assessed using accuracy scores, confusion matrices, and classification reports. Feature importance analysis is carried out to identify the most influential variables contributing to income prediction.

To address class imbalance, SMOTE is applied to the training data. Hyperparameter tuning is then performed using RandomizedSearchCV to optimize model performance. The improved model is evaluated again using the same metrics, and updated feature importance plots are generated to compare the results before and after optimization.

The repository presents a complete machine learning pipeline, covering data preprocessing, exploratory analysis, model training, imbalance handling, tuning, and evaluation using Python, Pandas, Scikit-learn, and visualization libraries.
