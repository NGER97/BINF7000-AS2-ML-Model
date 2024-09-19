import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# Load the dataset
data = pd.read_csv('G12_breast_dna-meth.csv', header=0, index_col=0)

# Remove any rows with missing values
data = data.dropna()

# Extract features (CpG sites) and the target variable (Label)
X = data.drop(columns=['Label'])  # Drop the 'Label' column, keeping all CpG sites as features
y = data['Label']  # Extract the 'Label' column as the target variable

# Map labels to numerical values, e.g., 'Primary Tumor' -> 0, 'Solid Tissue Normal' -> 1
y = y.map({'Primary Tumor': 0, 'Solid Tissue Normal': 1})

# Check for any unmapped labels and remove them
if y.isnull().any():
    print("Unmapped label values exist:")
    print(data[y.isnull()])
    # Drop unmapped labels
    y = y.dropna()
    X = X.loc[y.index]

# Ensure that all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Handle any NaN values that may have been introduced during conversion
X = X.fillna(X.mean())

# Standardize the features (Logistic Regression is sensitive to feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
log_reg = LogisticRegression(solver='liblinear', random_state=42)

# Define the hyperparameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength (lower values = stronger regularization)
    'penalty': ['l1', 'l2'],  # Regularization method: L1 (Lasso) or L2 (Ridge)
    'class_weight': [None, 'balanced']  # Handle class imbalance (optional)
}

# Set up GridSearchCV for hyperparameter optimization
grid_search = GridSearchCV(
    estimator=log_reg,  # Base estimator is logistic regression
    param_grid=param_grid,  # Hyperparameter grid
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Optimization based on accuracy
    n_jobs=-1,  # Use all available CPU cores for parallelization
    verbose=2  # Display progress during hyperparameter tuning
)

# Fit the model on the training data with hyperparameter search
grid_search.fit(X_train, y_train)

# Output the best hyperparameters found
print("Best parameters found:", grid_search.best_params_)

# Train the final model using the best hyperparameters
best_log_reg = grid_search.best_estimator_

# Save the best model to a file
joblib.dump(best_log_reg, 'logistic_regression_model.pkl')
print("Best model saved to 'logistic_regression_model.pkl'")

# Make predictions on the test set
y_pred = best_log_reg.predict(X_test)

# Print the classification report (accuracy, precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))
