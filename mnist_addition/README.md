# MNIST Addition

This repository contains code for a project that performs addition on MNIST digit images using neural networks.

## Project Structure

- `data/`: Contains the dataset and data loading scripts.
- `models/`: Contains the neural network models.
- `notebooks/`: Jupyter notebooks for experimentation and visualization.
- `scripts/`: Scripts for training and evaluating models.
- `utils/`: Utility functions and helper scripts.

## Files in `utils` Directory
- `visualize_weights.py`: Functions for visualizing the weights of the neural network.
  - `plot_weight_distributions(save_path)`: Plots the distribution of weights for each layer.
  - `plot_weight_statistics(save_path)`: Plots statistical measures of weights across layers.
  - `plot_weight_heatmaps(save_path)`: Plots heatmaps of weight matrices for each layer.
  - `generate_weight_summary()`: Prints summary statistics for model weights.
- `visualize_tuning.py`: Functions for visualizing hyperparameter tuning results.
  - `plot_parameter_distributions(save_path)`: Plots distributions of hyperparameters and their relationship with accuracy.
  - `plot_parameter_importance(save_path)`: Plots correlation between parameters and accuracy.
  - `plot_epochs_analysis(save_path)`: Plots relationship between epochs and performance.
  - `generate_summary_report()`: Prints a summary report of the hyperparameter tuning results.
- `t-sne_visualization.py`: Functions for visualizing high-dimensional data using t-SNE.
  - `extract_features_and_visualize(model, dataloader, save_dir, device)`: Extracts features and creates t-SNE visualizations.
- `run_inference.py`: Functions for running inference on trained models.
  - `load_model_and_run_inference(exp_dir, test_dataset, device)`: Loads a model and runs inference on test data.
  - `visualize_results(results, save_dir)`: Creates visualizations of inference results.
  - `run_all_experiments(base_dir, test_dataset, device)`: Runs inference on all experiment models.
- `plot_history.py`: Functions for plotting training history.
  - `plot_training_history(json_path)`: Plots training and validation metrics from metrics history JSON file.
- `delete_results.py`: Script for deleting result files.
  - `delete_results_files(tuning_dir)`: Deletes all results.json files from experiment directories.
- `compare_models.py`: Functions for comparing different models.
  - `train_and_evaluate_models(train_data, train_labels, test_data, test_labels, save_dir)`: Trains and evaluates Random Forest and SVM models.
  - `print_results(results)`: Prints detailed results for each model.
- `compare_dataset.py`: Functions for comparing balanced and unbalanced datasets.
  - `evaluate_model(model, test_loader, device)`: Evaluates model on test dataset.
  - `plot_comparison(balanced_metrics, unbalanced_metrics, save_path)`: Plots training history comparison.
- `prepare_sklearn_dataset.py`: Functions for preparing datasets for sklearn models.
  - `read_idx(path)`: Reads IDX file format.
  - `load_mnist(path)`: Loads MNIST data from raw files.
  - `create_digit_indices(labels)`: Creates mapping from digit to indices.
  - `generate_balanced_pairs(data, labels, digit_indices, pairs_per_sum)`: Generates balanced pairs for addition.
  - `prepare_datasets(mnist_path, save_path, seed)`: Prepares balanced datasets.
- `prepare_classification_dataset.py`: Functions for preparing classification datasets.
  - `read_idx(path)`: Reads IDX file format.
  - `load_mnist(path)`: Loads MNIST data from raw files.
  - `create_digit_indices(labels)`: Creates mapping from digit to indices.
  - `generate_balanced_pairs(data, labels, digit_indices, pairs_per_sum)`: Generates balanced pairs for addition.
  - `prepare_datasets(mnist_path, save_path, seed)`: Prepares balanced datasets.
- `extract_best_results.py`: Functions for extracting the best results from experiment tuning directories.
  - `count_epochs(experiment_path)`: Counts the number of epoch directories in an experiment.
  - `extract_experiment_results(experiment_path)`: Extracts the best model results from a single experiment directory.
  - `compile_all_results(tuning_dir)`: Compiles results from all experiments in the tuning directory.
- `compare_weak_classifier.py`: Functions for comparing weak classifiers.
  - `train_combined_classifier(X_train, y_train, X_test, y_test, n_samples)`: Trains a combined classifier.
  - `train_sequential_classifier(X_train_left, y_train_left, X_train_right, y_train_right, X_test_left, X_test_right, y_test, n_samples)`: Trains a sequential classifier.
  - `prepare_data_splits(data_path)`: Prepares data splits for training and testing.
  - `plot_results(results, save_path)`: Plots the results of classifier performance.
  - `compare_models(data_path, sample_sizes)`: Compares models based on different sample sizes.

## Getting Started

### Prerequisites

- Python 3.x
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
