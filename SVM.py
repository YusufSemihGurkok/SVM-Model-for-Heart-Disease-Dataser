import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
df = pd.read_csv('heart_disease.csv')
continuous_vars = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
categorical_vars = ['Gender', 'education', 'currentSmoker', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes']

df['Heart_ stroke'].replace({'No': 0, 'yes': 1}, inplace=True)
df['education'].replace({'uneducated': 0, 'primaryschool': 1, 'postgraduate': 2, 'graduate': 3}, inplace=True)
df.fillna(df.mean(), inplace=True)

def chi_square_of_df_cols(df, col1, col2):
    df_col1, df_col2 = df[col1], df[col2]

    result = [[sum((df_col1 == df_col1_unique) & (df_col2 == df_col2_unique))
               for df_col2_unique in df_col2.unique()]
              for df_col1_unique in df_col1.unique()]

    observed = pd.DataFrame(result, index=df_col1.unique(), columns=df_col2.unique())
    expected = df_col1.value_counts().values.reshape(-1, 1) * df_col2.value_counts().values / len(df)

    return ((observed - expected) ** 2 / expected).values.sum()

# calculate and print Pearson correlation for continuous variables
for var in continuous_vars:
    correlation = np.corrcoef(df[var], df['Heart_ stroke'])[0, 1]
    print(f"Pearson correlation between {var} and heartstroke: {correlation}")

# calculate and print Chi-square for categorical variables
for var in categorical_vars:
    chi_square_stat = chi_square_of_df_cols(df, var, 'Heart_ stroke')
    print(f"Chi-square statistic between {var} and heartstroke: {chi_square_stat}")

# Correlation matrix heatmap for all features
plt.figure(figsize=(15,10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap=plt.cm.Reds)
plt.show()

# Define the SVM class
class SVM:
    def __init__(self, learning_rate=0.001, num_iterations=100, regularization_param=0.01):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization_param = regularization_param
        self.weights = None
        self.bias = None

    def train(self, X, y):
        num_samples, num_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Gradient descent optimization
        for _ in range(self.num_iterations):
            linear_output = np.dot(X, self.weights) + self.bias

            # Apply hinge loss function
            hinge_loss = np.maximum(0, 1 - y * linear_output)

            # Compute gradients
            dW = (self.regularization_param * self.weights) - np.sum(X.T * (y * (hinge_loss > 0)), axis=1) / num_samples
            db = -np.sum(y * hinge_loss > 0) / num_samples

            # Update weights and bias
            self.weights -= self.learning_rate * dW
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)

    
def pca(X, k, change_inner=False):
        # Mean center the data
        X = X - np.mean(X, axis=0)

        # Calculate the covariance matrix
        cov = np.cov(X, rowvar=False)

        # Calculate the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the top k eigenvectors
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:k]
        eigenvectors = eigenvectors[:, idx][:, :k]

         # Project the data onto the eigenvectors
        X_transformed = X @ eigenvectors

        return X_transformed
# One-hot encoding function
def one_hot_encode(df, column):
    unique_values = df[column].unique()
    for value in unique_values:
        df[column+'_'+str(value)] = (df[column] == value).astype(int)
    df.drop(column, axis=1, inplace=True)
    return df

# Cross validation function
def cross_validation(svm, X, y, K=5):
    fold_size = len(X) // K
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(K):
        X_train, X_val = np.concatenate((X[:i*fold_size], X[(i+1)*fold_size:])), X[i*fold_size:(i+1)*fold_size]
        y_train, y_val = np.concatenate((y[:i*fold_size], y[(i+1)*fold_size:])), y[i*fold_size:(i+1)*fold_size]

        # Train the model and make predictions
        svm.train(X_train, y_train)
        y_pred = svm.predict(X_val)

        # Compute metrics manually
        tp = np.sum((y_val == 1) & (y_pred == 1))
        tn = np.sum((y_val == 0) & (y_pred == 0))
        fp = np.sum((y_val == 0) & (y_pred == 1))
        fn = np.sum((y_val == 1) & (y_pred == 0))

        # Precision
        precision = tp / (tp + fp)
        precisions.append(precision)

        # Recall
        recall = tp / (tp + fn)
        recalls.append(recall)

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

# Load the dataset
df = pd.read_csv('heart_disease.csv')

# Fill missing values with column means
df.fillna(df.mean(), inplace=True)

# Encode categorical variables using one-hot encoding
categorical_columns = ['Gender', 'education', 'prevalentStroke', 'Heart_ stroke']
for column in categorical_columns:
    df = one_hot_encode(df, column)

# Split the dataset into features (X) and target variable (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Fill missing values in the training and test sets
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train))
X_test = pd.DataFrame(imputer.transform(X_test))

# Perform PCA
X_train_pca = pca(X_train, k=10, change_inner=True)
X_test_pca = pca(X_test, k=10)

# Train the SVM classifier and apply cross-validation
accuracy, precision, recall, f1_score = cross_validation(svm, X_train_pca, y_train, K=5)
print("Cross Validation Accuracy: ", accuracy)
print("Cross Validation Precision: ", precision)
print("Cross Validation Recall: ", recall)
print("Cross Validation F1 Score: ", f1_score)