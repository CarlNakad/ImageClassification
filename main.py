import os
from datetime import datetime
from dataset import load_or_cache_data
from evaluate_models import evaluate_model, generate_comparison_table
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import torch
import argparse
from model_runner import ModelRunner

MODEL_DATA_PATH = "./data/model_data"

if __name__ == '__main__':
    # Create a folder to save the figures
    date_string = datetime.now().strftime("%d-%m-%Y %H:%M")
    FIGURE_PATH = f'./figures/{date_string}'
    print(f"Saving figures to {FIGURE_PATH}")
    os.makedirs(FIGURE_PATH, exist_ok=True)

    # Create a folder to save the model data
    os.makedirs(MODEL_DATA_PATH, exist_ok=True)

    # Set the device to GPU if available (mps is used for Apple Silicon processors)
    # This has only been tested on mps since I own a macbook
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()  else "cpu")

    # Load or cache the data
    train_feature, train_labels, test_features, test_labels = load_or_cache_data(batch_size=32, device=device)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    model_runner = ModelRunner(train_feature, train_labels, test_features, test_labels, classes, FIGURE_PATH, device)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the CIFAR-10 project.")
    parser.add_argument("--model", choices=["naive_bayes", "decision_tree", "mlp", "cnn", "all"], default="all",
                        help="Specify which model to run.")
    parser.add_argument("--max-depth", type=int, default=50, help="Specify the max depth for the decision tree model.")
    parser.add_argument("--epochs", type=int, default=50, help="Specify the number of epochs for the MLP model or CNN model.")
    parser.add_argument("--hidden-layer-size", type=int, default=512, help="Specify the hidden layer size for the MLP model.")
    parser.add_argument("--layer-removal", action="store_true", default=False, help="Specify whether to remove a layer from the MLP model or CNN model.")
    parser.add_argument("--layer-addition", action="store_true", default=False, help="Specify whether to add a layer to the MLP model or CNN model.")
    parser.add_argument("--kernal-size", type=int, default=3, help="Specify the kernel size for the CNN model.")
    args = parser.parse_args()

    if args.model == "naive_bayes":
        # Train and evaluate using my Gaussian Naive Bayes implementation
        model_runner.run_naive_bayes_model()
    elif args.model == "decision_tree":
        # Train and evaluate using my Decision Tree implementation
        depth = args.max_depth
        model_runner.run_decision_tree_model(depth)
    elif args.model == "mlp":
        # Train and evaluate using my Multi Layer Perceptron implementation
        num_epochs = args.epochs
        hidden_layer_size = args.hidden_layer_size
        layer_removal = args.layer_removal
        layer_addition = args.layer_addition
        model_runner.run_mlp_model(num_epochs, hidden_layer_size, layer_removal, layer_addition)
    elif args.model == "cnn":
        # Train and evaluate using my Convolution Neural Network implementation
        num_epochs = args.epochs
        kernel_size = args.kernal_size
        layer_removal = args.layer_removal
        layer_addition = args.layer_addition
        model_runner.run_cnn_model(num_epochs, kernel_size, layer_removal, layer_addition)
    else:
        # Train and evaluate using my Gaussian Naive Bayes implementation
        model_runner.run_naive_bayes_model()

        # Train and evaluate using sklearn's Gaussian Naive Bayes implementation
        print("Training and evaluating sklearn's Gaussian Naive Bayes model...")
        model = GaussianNB()
        test_pred = model.fit(train_feature, train_labels).predict(test_features)
        model_runner.append_report(evaluate_model("SK GaussianNB model", test_labels, test_pred, classes, FIGURE_PATH))

        # Train and evaluate using my Decision Tree implementation with different depths
        decision_tree_depths = [1, 5, 10, 15, 50]
        for depth in decision_tree_depths:
            model_runner.run_decision_tree_model(depth)

        # Train and evaluate using sklearn's Decision Tree implementation
        print("Training and evaluating sklearn's Decision Tree model with depth 50...")
        model = DecisionTreeClassifier(max_depth=50)
        test_pred = model.fit(train_feature, train_labels).predict(test_features)
        model_runner.append_report(evaluate_model("SK Decision Tree model with depth 50",test_labels, test_pred, classes, FIGURE_PATH))

        # Train and evaluate using my Multi Layer Perceptron implementation
        num_epochs = 50
        hidden_layer_sizes = [256, 512, 1024]
        for hidden_layer_size in hidden_layer_sizes:
            model_runner.run_mlp_model(num_epochs, hidden_layer_size, False, False)

        model_runner.run_mlp_model(num_epochs, 512, True, False)
        model_runner.run_mlp_model(num_epochs, 512, False, True)

        # Train and evaluate using my Convolution Neural Network implementation
        num_epochs = 50
        kernel_sizes = [3, 5]
        for kernel_size in kernel_sizes:
           model_runner.run_cnn_model(num_epochs, kernel_size, False, False)

        model_runner.run_cnn_model(num_epochs, 3, True, False)
        model_runner.run_cnn_model(num_epochs, 3, False, True)

    # Create table of models' classification reports
    print("\nGenerating comparison table...")
    generate_comparison_table(model_runner.models_report, FIGURE_PATH)


