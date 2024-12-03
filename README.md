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
│   ├── test1/                # Test experiment 1 (balanced)
│   ├── test2/                # Test experiment 2 (unbalanced)
│   └── tuning_20241127_201445/ # Hyperparameter tuning experiment (containing results for tnse, weight visualization)
├── mnist_addition/           # Main project directory
│   ├── __init__.py           # Init file for the package
│   ├── config.toml           # Configuration file
│   ├── data/                 # Data related scripts and utilities
│   ├── [main.py]               # Main script to run the project
│   ├── [main_tuning.py]       # Script for hyperparameter tuning
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
   pip install -r requirements.txt

## Usage
### Running with the main script

To run the main script of training the model, execute:
    ```bash
    python main.py
    ```

## Hyperparameter Tuning

To perform hyperparameter tuning, execute (may take 12hs):
    ```bash
    python main_tuning.py
    ```

### Visualizing the Dataset
To visualize the dataset, open the Jupyter notebook:
    ```bash
    jupyter notebook mnist_addition/visualize_dataset.ipynb
    ```

