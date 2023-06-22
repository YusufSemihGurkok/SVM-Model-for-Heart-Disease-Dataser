# Heart Disease Prediction using SVM

This project focuses on predicting the risk of heart disease based on various parameters using Support Vector Machines (SVM). The dataset consists of parameters such as gender, age, education, smoking habits, blood pressure, cholesterol levels, BMI, and glucose levels.

## Dataset

The dataset used in this project is provided in the file `heart_disease.csv`. It contains the following parameters:

- Gender: Categorical (Male/Female)
- Age: Continuous
- Education: Categorical (Uneducated/Primary School/Other)
- Current Smoker: Categorical (0: No, 1: Yes)
- Cigarettes Per Day: Continuous
- BP Meds: Categorical (0: No, 1: Yes)
- Prevalent Stroke: Categorical (No/Yes)
- Prevalent Hypertension: Categorical (0: No, 1: Yes)
- Diabetes: Categorical (0: No, 1: Yes)
- Total Cholesterol: Continuous
- Systolic Blood Pressure: Continuous
- Diastolic Blood Pressure: Continuous
- BMI: Continuous
- Heart Rate: Continuous
- Glucose: Continuous
- Heart Stroke: Categorical (No/Yes)

## How it works

1. The dataset is loaded from the file `heart_disease.csv`. It contains information about various parameters and the occurrence of heart disease.

2. The dataset is preprocessed to handle missing values. Missing values are filled with the column means.

3. Categorical variables are encoded using one-hot encoding to represent them as binary features. The categorical variables include gender, education, prevalent stroke, and heart stroke.

4. The dataset is split into features (X) and the target variable (y). X contains all the parameters except the 'Heart_ stroke' column, while y contains the 'Heart_ stroke' column indicating the occurrence of heart disease.

5. The data is further split into training and test sets using the train_test_split function from scikit-learn. The test set will be used to evaluate the final SVM model.

6. Missing values in the training and test sets are filled with column means using the SimpleImputer class from scikit-learn.

7. Principal Component Analysis (PCA) is performed on the training set to reduce the dimensionality of the data. PCA helps capture the most important features and can improve the performance of the SVM model.

8. The SVM model is trained using the training set, and k-fold cross-validation is applied to evaluate its performance. The dataset is divided into k folds, and the model is trained on k-1 folds while evaluating the performance on the remaining fold. This process is repeated k times, and the average accuracy, precision, recall, and F1 score are calculated.

9. Finally, the trained SVM model is evaluated on the test set, and the accuracy, precision, recall, and F1 score are calculated.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
