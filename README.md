[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

# Image and Tabular Data Classification with PyTorch-Lightning

## Overview

This project demonstrates the integration of **tabular classification**, **image classification**, and **custom dataset handling** using PyTorch, PyTorch Lightning, and Scikit-learn. The script includes:

1. Training a tabular classifier on synthetic data using Scikit-learn's `RandomForestClassifier`.
2. Training an image classifier on custom datasets or synthetic datasets using PyTorch Lightning.
3. Leveraging custom dataset loaders for seamless integration of custom data pipelines.

---

## File Structure

- **`main.py`**: The main entry point that runs the tabular and image classification tasks.
- **`plmodels.py`**: Contains the implementation of:
  - `TabularClassifier`: For tabular data classification.
  - `ImageClassifier`: For image classification using pre-trained models like ResNet.
  - `ConvAutoEncoder`: For autoencoder-based tasks.
- **`GenericDataset.py`**: Includes classes for handling:
  - Custom datasets (`CustomDataset`).
  - CSV-based datasets (`CustomCSVDataset`).
  - Generic dataset loader (`GenericDatasetLoader`).

---

## Requirements

- Python 3.8 or higher
- Required Libraries:
  ```bash
  pip install torch torchvision pytorch-lightning scikit-learn numpy pandas

## How to Run
To execute the program, run the following command:
```bash
python main.py
```

# Image Classification

This project provides a framework for image classification using a pre-trained ResNet18 model. It supports popular datasets and allows for custom dataset integration.

## Features

- **Image Loading**:
  - Uses the `GenericDatasetLoader` class to load images.
  - Supports:
    - CIFAR10
    - MNIST
    - FLOWER102
    - Custom datasets (via directories or CSV files).

- **Training**:
  - Trains an `ImageClassifier` based on a pre-trained ResNet18 model.

- **Outputs**:
  - Training logs with metrics:
    - Accuracy
    - Loss

## How to Use

1. Load datasets with `GenericDatasetLoader`.
2. Train the classifier using the provided ResNet18 architecture.
3. Monitor training logs for performance metrics.

Enjoy high-performance image classification with ease!

## Author

- [@fahadahmedkhokhar](https://www.github.com/fahadahmedkhokhar)

## License

[MIT](https://choosealicense.com/licenses/mit/)

