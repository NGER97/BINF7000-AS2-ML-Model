# BINF7000-AS2-ML-Model

## Data

My file structure:

```sh
├── G12_breast_dna-meth.csv
├── G12_breast_gene-expr.csv
├── README.md
├── dna_meth_rf.py
├── dna_meth_lr.py
├── dna_meth_fnn.py
├── logistic_regression_model.pkl
├── ffnn_model.keras
└── random_forest_model.pkl
```

## Workflow

- Data Loading and Preprocessing: The code reads data from a CSV file, handles missing values, and ensures numeric features are ready for training.
- Feature and Label Extraction: It extracts the CpG sites as features and the target labels, mapping them into numeric values for classification.
- Train-Test Split: The dataset is split into training and testing sets to evaluate the model's performance later.
- Hyperparameter Optimization: A grid search is used to find the best hyperparameters for the random forest model through cross-validation.
- Feature Importance Analysis: After training, the feature importances are extracted and analyzed to identify the most influential CpG sites.
- Visualization: The top 10 most important features are visualized using a bar chart for better interpretation.
- Prediction and Evaluation: The trained model is tested on the unseen data (test set), and performance metrics are printed.
- Model Saving: The trained model is saved to a file for future use or deployment.

## Ranmdon Forest in DNA_meth

```
Best parameters: {'max_depth': None, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
```

```
Top 10 most important features:
              CpG  Importance
3383  cg18362003    0.012105
1406  cg07489502    0.011681
4883  cg27393010    0.006163
1518  cg08104202    0.006125
2505  cg13539545    0.005845
3396  cg18438793    0.005843
3022  cg16348470    0.005814
4685  cg26299169    0.005395
1812  cg09513990    0.005326
3945  cg21884231    0.005221
```

```
Classification Report:
               precision    recall  f1-score   support

           0       0.99      1.00      1.00       160
           1       1.00      0.94      0.97        16

    accuracy                           0.99       176
   macro avg       1.00      0.97      0.98       176
weighted avg       0.99      0.99      0.99       176
```

## Logistic Regression in DNA_meth

```
Best parameters found: {'C': 1, 'class_weight': 'balanced', 'penalty': 'l1'}
```

```
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.99      1.00       160
           1       0.94      1.00      0.97        16

    accuracy                           0.99       176
   macro avg       0.97      1.00      0.98       176
weighted avg       0.99      0.99      0.99       176
```

## Feedforward Neural Network in DNA_meth

```
Epoch 50/50
18/18 ━━━━━━━━━━━━━━━━━━━━ 0s 5ms/step - accuracy: 1.0000 - loss: 4.5866e-06 - val_accuracy: 0.9929 - val_loss: 0.0684
```

```
Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.97      0.98       160
           1       0.76      1.00      0.86        16

    accuracy                           0.97       176
   macro avg       0.88      0.98      0.92       176
weighted avg       0.98      0.97      0.97       176
```

## TODO

- Further optimization of hyperparameter optimization
- I have tried to use the method of balancing data categories, but the trained model is not ideal and I have not found a better solution.
- Feature engineering and dimensionality reduction are needed or not?
