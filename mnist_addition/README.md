# MNIST Addition

This repository contains code for a project that performs addition on MNIST digit images using neural networks.

## Project Structure

- `data/`: Contains the dataset and data loading scripts.
- `models/`: Contains the neural network models.
- `training/`: Contains scripts and configurations for training the models.
- `utils/`: Utility functions and helper scripts.

## Files in `utils` Directory
- `visualize_weights.py`: Functions for visualizing the weights of the neural network.
- `visualize_tuning.py`: Functions for visualizing hyperparameter tuning results.
- `t-sne_visualization.py`: Functions for visualizing high-dimensional data using t-SNE.
- `run_inference.py`: Functions for running inference on trained models.
- `plot_history.py`: Functions for plotting training history.
- `delete_results.py`: Script for deleting result files.
- `compare_models.py`: Functions for comparing different models.
- `compare_dataset.py`: Functions for comparing balanced and unbalanced datasets.
- `prepare_sklearn_dataset.py`: Functions for preparing datasets for sklearn models.
- `prepare_classification_dataset.py`: Functions for preparing classification datasets.
- `extract_best_results.py`: Functions for extracting the best results from experiment tuning directories.
- `compare_weak_classifier.py`: Functions for comparing weak classifiers.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

 Clone the repository

### Note

1. I have cleared the results in experinments using `delete_results.py`. If we run again, we might get nothing.
2. Some results that I have show in the report is obtained by running the scripts, and the results are not stored properly.
3. Note that we couldn't run `run_inference.py` as not all model.pt is uploaded to gitlab.

### Results Reproduction

1. **Compare the weak classifiers**:
    ```sh
    python compare_weak_classifier.py
    ```
2. **Compare the SVM, RF, NN**:
    ```sh
    python compare_models.py
    ```
3. **tsne results**:
    ```sh
    python t-sne_visualization.py
    ```
3. **visualize weights of best model**:
    ```sh
    python  visualize_weights.py
    ```
4. **visualize finetuning results**:
    ```sh
    python  visualize_tuning.py
    ```
5. **extract best performing finetuned results**:
    ```sh
    python  extract_best_results.py
    ```
6. **extract best performing epoch in one training**:
    ```sh
    python  plot_history.py
    ```



