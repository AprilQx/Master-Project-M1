# M1 Machine Learning Coursework - MNIST Addition

This repository contains the implementation of a machine learning pipeline for adding two handwritten MNIST digits. The project explores various approaches including neural networks, traditional ML algorithms, and weak classifiers.

## Project Structure
```bash
.
├── data/                      # Data handling and preprocessing
│   ├── MNIST/                # Raw MNIST dataset(# Auto-downloaded, not in Git)
│   ├── processed/            # Processed dataset for addition task (# Generated data, not in Git)
├── experiments/              # Experimental results
│   ├── sklearn_comparison/   # Comparison experiments with sklearn models
│   ├── test1/                # Test experiment 1 (balanced, containing best_model.pt) 
│   ├── test2/                # Test experiment 2 (unbalanced, containing best_model.pt)
│   ├── tuning_20241127_201445/ # Hyperparameter tuning experiment (grid search)
│   ├── weak_classifier_results #sequential and combined comparison
│   ├──optuna_20241204_164643 #optuna fine tuning results (containing best_model.pt in trial 11)
│   └──  training_comparison.png #comparison between training process of test1 and test2
├── mnist_addition/           # Main project directory
│   ├── __init__.py           # Init file for the package
│   ├── config.toml           # Configuration file
│   ├── data/                 # Data related scripts and utilities
│   ├── [main.py]               # Main script to run the project
│   ├── [main_tuning.py]       # Script for hyperparameter tuning (grid search/abandoned!!)
│   ├── [tuning_optuna.py]   #script for hyperparameter tuning using optuna
│   ├── models/               # Model implementations
│   ├── [README.md]            # Project README
│   ├── training/             # Training procedures
│   ├── utils/                # Utility functions
│   └── visualize_dataset.ipynb # Notebook for dataset visualization
├── report/                   # Reports and documentation
├── requirements.txt          # Project dependencies
└── [README.md] 
```

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt

## Usage
### Running with the main script

To run the main script of training the model, execute:
    ```
    python main.py
    ```

### Hyperparameter Tuning

To perform hyperparameter tuning, execute (may take 6hs):
    ```
    python tuning_optuna.py
    ```

### Visualizing the Dataset
To visualize the dataset, open the Jupyter notebook:
    ```
    jupyter notebook mnist_addition/visualize_dataset.ipynb
    ```

