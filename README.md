# CIFAR-10 Image Classification Project

## Project Overview

This project implements image classification using CIFAR-10 dataset. The machine learning models used in this project are:
- Gaussian Naive Bayes
- Decision Tree
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)

This project also includes pre-processing and evaluation pipelines to analyze model performance.

## Project Contents

| File Name                         | Description                                                                                                                                                                     |
|-----------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `main.py`                         | Entry point to train and evaluate all models. It also takes care of loading the dataset.                                                                                        |
| `dataset.py`                      | Handles dataset loading, pre-processing, feature extraction, PCA, as well as saving or loading cached filtered dataset.                                                         |
| `run_model.py`                    | Contains code for training and evaluating individual models.                                                                                                                    |
| `evalulate_models.py`             | Contains code for evaluating models using confusion matrix and calculates the precision, accuracy, recall, and F1-measure which is then plotted in a table with the model name. |
| `models/gaussian_naive_bayes.py`  | Implements Gaussian Naive Bayes model.                                                                                                                                          |
| `models/common/tree_node.py`      | Contains the TreeNode class which is used to build the decision tree.                                                                                                           |
| `models/decision_tree.py`         | Implements Decision Tree classifier with depth variants.                                                                                                                        |
| `models/common/neural_network.py` | Base class for neural network models (MLP and CNN)                                                                                                                              |
| `models/mlp.py`                   | Implements the Multi-Layer Perceptron (MLP) with additional parameters to add/remove layers and vary hidden layer sizes.                                                        |
| `models/cnn.py`                   | Implements the Convolution Neural Network (CNN) with VGG11 Architecture. It support additional parameters such as add/remove layers and kernel size.                            |

## Steps to Execute the Project

### Step 1: Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 2: Execute the Main Script

The main.py script supports several command-line arguments to specify the models and their variants to train and evaluate.
To view all the available options, run the following command:

```bash
python main.py --help
```

#### Usage

```bash
python main.py [--model MODEL] [--max-depth MAX_DEPTH] [--epochs EPOCHS] [--hidden-layer-size HIDDEN_LAYER_SIZE] [--layer-removal] [--layer-addition] [--kernal-size KERNAL_SIZE]
```

#### Arguments:

| Argument            | Default Value | Description                                                                           |
|---------------------|---------------|---------------------------------------------------------------------------------------|
| --model             | all           | Specify model to run and evaluate. Options: all, naive_bayes, decision_tree, mlp, cnn |
| --max-depth         | 10            | Maximum depth of the decision tree.                                                   |
| --epochs            | 10            | Number of trainings epochs for the MLP and CNN models.                                |
| --hidden-layer-size | 512           | Hidden layer size in the MLP model.                                                   |
| --layer-removal     | False         | Flag to remove a layer from the MLP or CNN model.                                     |
| --layer-addition    | False         | Flag to add a layer to the MLP or CNN model.                                          |
| --kernal-size       | 3             | Kernel size in the CNN model.                                                         |

#### Example Commands

#### 1. To train and evaluate all models:

```bash
python main.py
```

#### 2. Run Gaussian Naive Bayes Model Only:

```bash
python main.py --model naive_bayes
```

#### 3. Run Decision Tree Model with custom depth:

```bash
python main.py --model decision_tree --max-depth 5
```

#### 4. Run MLP Model with custom hidden layer size and epochs:

```bash
python main.py --model mlp --hidden-layer-size 256 --epochs 20
```

#### 5. Run CNN Model with layer addition and custom kernel size and epochs:

```bash
python main.py --model cnn --layer-addition --kernal-size 3 --epochs 20
```
