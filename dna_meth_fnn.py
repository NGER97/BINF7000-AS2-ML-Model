import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras import models, layers
import keras_tuner as kt  # For hyperparameter tuning

# Load the dataset
data = pd.read_csv('G12_breast_dna-meth.csv', header=0, index_col=0)

# Remove any rows with missing values
data = data.dropna()

# Extract features (CpG sites) and the target variable (Label)
X = data.drop(columns=['Label'])  # Drop the 'Label' column, keeping all CpG sites as features
y = data['Label']  # Extract the 'Label' column as the target variable

# Map labels to numerical values, e.g., 'Primary Tumor' -> 0, 'Solid Tissue Normal' -> 1
y = y.map({'Primary Tumor': 0, 'Solid Tissue Normal': 1})

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Handle any NaN values (fill with column means)
X = X.fillna(X.mean())

# Standardize the features (Neural Networks benefit from scaled inputs)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a function to build the feedforward neural network model
def build_model(hp):
    model = models.Sequential()
    # Add an input layer
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

    # Tune the number of layers
    for i in range(hp.Int('num_layers', 1, 3)):
        # Tune the number of units per layer
        model.add(layers.Dense(units=hp.Int(f'units_{i}', min_value=32, max_value=512, step=32),
                               activation='relu'))

    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),  # Tune learning rate
        loss='binary_crossentropy',  # Binary classification problem
        metrics=['accuracy']
    )
    return model


# Set up KerasTuner for hyperparameter tuning
tuner = kt.Hyperband(
    build_model,  # Function that builds the model
    objective='val_accuracy',  # Objective to optimize (validation accuracy)
    max_epochs=10,  # Number of epochs to run each model configuration
    factor=3,  # Hyperband factor
    directory='hyperband_tuning',  # Where to store tuning results
    project_name='breast_dna_methylation'
)

# Perform hyperparameter search
tuner.search(X_train, y_train, epochs=10, validation_split=0.2)

# Get the best model after tuning
best_model = tuner.get_best_models(num_models=1)[0]

# Train the best model
history = best_model.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = best_model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Make predictions and print the classification report
y_pred = (best_model.predict(X_test) > 0.5).astype(int)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the best model to a file
best_model.save('ffnn_model.keras')
print("Best model saved to 'ffnn_model.h5'")
