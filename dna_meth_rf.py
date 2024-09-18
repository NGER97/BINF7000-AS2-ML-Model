import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib

# Read the CSV file
data = pd.read_csv('G12_breast_dna-meth.csv', header=0, index_col=0)

# Remove rows that contain missing values
data = data.dropna()

# Use the CpG site columns as features (X) and 'Label' as the target variable (y)
X = data.drop(columns=['Label'])  # Remove the 'Label' column, keeping all CpG sites
y = data['Label']  # Extract the 'Label' column as the target

# Convert labels to numerical encoding, e.g., 'Primary Tumor' -> 0, 'Solid Tissue Normal' -> 1
y = y.map({'Primary Tumor': 0, 'Solid Tissue Normal': 1})

# Check for unmapped labels
if y.isnull().any():
    print("Unmapped label values exist:")
    print(data[y.isnull()])
    # Remove unmapped labels
    y = y.dropna()
    X = X.loc[y.index]

# Ensure all features are numerical
X = X.apply(pd.to_numeric, errors='coerce')

# Handle any NaNs that may have been introduced during conversion
X = X.fillna(X.mean())  # Fill NaNs with the mean of each column

# Split the dataset into training and testing sets, 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,  # Use all available CPU cores
    scoring='accuracy',  # Evaluation metric
    verbose=2  # Display the progress of hyperparameter tuning
)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Output the best parameters found
print("Best parameters:", grid_search.best_params_)

# Train the final model using the best parameters
best_rf_model = grid_search.best_estimator_

# Extract feature importances
importances = best_rf_model.feature_importances_

# Bind feature importances with feature names
feature_importance = pd.DataFrame({'CpG': X.columns, 'Importance': importances})

# Sort by importance
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Display the top 10 most important features
print("Top 10 most important features:\n", feature_importance.head(10))

# Visualize the top 10 most important CpG sites
plt.figure(figsize=(10, 6))
plt.title('Top 10 CpG Sites by Feature Importance')
plt.barh(feature_importance['CpG'][:10], feature_importance['Importance'][:10], color='b')
plt.xlabel('Importance')
plt.gca().invert_yaxis()  # Display in descending order
plt.show()

# Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Print the classification report (accuracy, precision, recall, F1-score)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(rf_model, 'random_forest_model.pkl')
