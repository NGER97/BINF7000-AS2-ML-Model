import matplotlib.pyplot as plt
import numpy as np

# Set the datasets and their labels
datasets = ['G12 Gene Expr', 'G12 DNA Meth', 'Mystery Gene Expr']
categories = ['Normal Tissue', 'Primary Tumour']
models = ['Logistic Regression', 'Random Forest', 'Neural Network']

# Using values provided from the table
precision_values = {
    'Normal Tissue': {
        'G12 Gene Expr': [0.84, 0.96, 0.81],
        'G12 DNA Meth': [0.86, 0.90, 0.86],
        'Mystery Gene Expr': [0.95, 1.00, 1.00]
    },
    'Primary Tumour': {
        'G12 Gene Expr': [1.00, 1.00, 1.00],
        'G12 DNA Meth': [1.00, 1.00, 1.00],
        'Mystery Gene Expr': [0.97, 0.97, 0.97]
    }
}
recall_values = {
    'Normal Tissue': {
        'G12 Gene Expr': [0.95, 1.00, 1.00],
        'G12 DNA Meth': [1.00, 1.00, 1.00],
        'Mystery Gene Expr': [0.97, 0.97, 0.97]
    },
    'Primary Tumour': {
        'G12 Gene Expr': [0.98, 1.00, 0.98],
        'G12 DNA Meth': [0.98, 0.99, 0.98],
        'Mystery Gene Expr': [0.95, 1.00, 1.00]
    }
}
f1_values = {
    'Normal Tissue': {
        'G12 Gene Expr': [0.89, 0.98, 0.90],
        'G12 DNA Meth': [0.93, 0.95, 0.93],
        'Mystery Gene Expr': [0.96, 0.99, 0.99]
    },
    'Primary Tumour': {
        'G12 Gene Expr': [0.99, 1.00, 0.99],
        'G12 DNA Meth': [0.99, 0.99, 0.99],
        'Mystery Gene Expr': [0.96, 0.99, 0.99]
    }
}

# Set up 3D sub-category layout and color scheme
width = 0.2  # The width of each bar
x = np.arange(len(models))  # The positions of the three models on the X-axis
colors = ['#a6cee3', '#b2df8a', '#fb9a99']  # Colors for each metric (light blue, light green, light red)

# Create the plot
fig, ax = plt.subplots(2, len(datasets), figsize=(15, 8), sharey=True)

# Draw bar charts for each dataset and category
for col_idx, dataset in enumerate(datasets):
    for row_idx, category in enumerate(categories):
        # Plot the bars for Precision, Recall, and F1-Score
        ax[row_idx, col_idx].bar(x - width, precision_values[category][dataset], width, label='Precision',
                                 color=colors[0], alpha=0.7)
        ax[row_idx, col_idx].bar(x, recall_values[category][dataset], width, label='Recall', color=colors[1], alpha=0.7)
        ax[row_idx, col_idx].bar(x + width, f1_values[category][dataset], width, label='F1-Score', color=colors[2],
                                 alpha=0.7)

        # Set axis labels
        ax[row_idx, col_idx].set_xticks(x)
        ax[row_idx, col_idx].set_xticklabels(models, rotation=45, ha='right')
        ax[row_idx, col_idx].set_ylim([0.8, 1.0])
        ax[row_idx, col_idx].set_title(f'{dataset} ({category})')

# Add a common legend
fig.suptitle('Performance Comparison of Models across Datasets and Classifications', fontsize=16)
fig.legend(['Precision', 'Recall', 'F1-Score'], loc='upper left')
fig.tight_layout()

# Show the plot
plt.show()

