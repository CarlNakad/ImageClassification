from dataset import load_cifar10
from evaluate_models import evaluate_model
from models.gaussian_naive_bayes import GaussianNaiveBayes
from models.decision_tree import DecisionTree
from models.cnn import ConvolutionNeuralNetwork
from models.mlp import MultiLayerPerceptron
import torchvision.transforms as transforms
MODEL_DATA_PATH = "./data/model_data"

class ModelRunner:
    def __init__(self, train_feature, train_labels, test_features, test_labels, classes, FIGURE_PATH, device):
        """
       Model Runner class to run and evaluate different models.

        :param train_feature: Dataset features for training
        :param train_labels: Dataset labels for training
        :param test_features: Dataset features for testing
        :param test_labels: Dataset labels for testing
        :param classes: Classes of the dataset
        :param FIGURE_PATH: Path to save the figures
        :param device: Device to run the models on
        """
        self.models_report = []
        self.train_feature = train_feature
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.classes = classes
        self.FIGURE_PATH = FIGURE_PATH
        self.device = device
        self.cnn_train_loader = None
        self.cnn_test_loader = None

    def append_report(self, report):
        """
        Append the report to the list of reports.

        :param report: Report to append
        :return: None
        """
        self.models_report.append(report)

    def run_naive_bayes_model(self):
        """
        Run the Gaussian Naive Bayes model and evaluate it.

        :return: None
        """

        print("Training and evaluating my GaussianNB model...")
        naive_bayes_model = GaussianNaiveBayes()
        test_pred = naive_bayes_model.fit(self.train_feature, self.train_labels).predict(self.test_features)
        self.append_report(evaluate_model("My GaussianNB model", self.test_labels, test_pred, self.classes, self.FIGURE_PATH))


    def run_decision_tree_model(self, max_depth):
        """
        Run the Decision Tree model and evaluate it.

        :param max_depth: Maximum depth of the decision tree
        :return: None
        """

        print(f"Training and evaluating my Decision Tree model with depth {max_depth}...")
        decision_tree_model = DecisionTree(max_depth, file_path=f"{MODEL_DATA_PATH}/decision_tree_model_{max_depth}.pkl")
        test_pred = decision_tree_model.fit(self.train_feature, self.train_labels).predict(self.test_features)
        self.append_report(
            evaluate_model(f"My Decision Tree model with depth {max_depth}", self.test_labels, test_pred, self.classes, self.FIGURE_PATH))


    def run_mlp_model(self, num_epochs, hidden_layer_size, layer_removal, layer_addition):
        """
        Run the Multi Layer Perceptron model and evaluate it.

        :param num_epochs: Number of epochs
        :param hidden_layer_size: Size of the hidden layer
        :param layer_removal: Whether to remove a layer
        :param layer_addition: Whether to add a layer
        :return: None
        """

        print(f"Training and evaluating my MLP model with {num_epochs} epochs and hidden layer size {hidden_layer_size}{' and removed layer' if layer_removal else ''}{' and additional layer' if layer_addition else ''}...")

        model_name = f"My MLP model with {num_epochs} epochs and hidden layer size {hidden_layer_size}{' and removed layer' if layer_removal else ''}{' and additional layer' if layer_addition else ''}"
        file_path = f"{MODEL_DATA_PATH}/bestMLP_model_{num_epochs}_{hidden_layer_size}{'_additional_layer' if layer_addition else ''}{'_removed_layer' if layer_removal else ''}.pth"

        mlp_model = MultiLayerPerceptron(device=self.device, num_epochs=num_epochs, hidden_layer_size=hidden_layer_size,
                                     layer_removal=layer_removal, layer_addition=layer_addition, file_path=file_path)
        test_pred = (mlp_model.fit(train_features=self.train_feature, train_labels=self.train_labels)
                     .predict(test_features=self.test_features, test_labels=self.test_labels))
        self.append_report(evaluate_model(model_name, self.test_labels, test_pred, self.classes, self.FIGURE_PATH))

    def run_cnn_model(self, num_epochs, kernel_size, layer_removal, layer_addition):
        """
        Run the Convolution Neural Network model and evaluate it.

        :param num_epochs: Number of epochs
        :param kernel_size: Kernel size
        :param layer_removal: Whether to remove a layer
        :param layer_addition: Whether to add a layer
        :return: None
        """

        print(f"Training and evaluating my CNN model with {num_epochs} epochs and kernel size {kernel_size}{' and removed layer' if layer_removal else ' and additional layer' if layer_addition else ''}...")
        file_path = f"{MODEL_DATA_PATH}/bestCNN_model_{num_epochs}_{kernel_size}{'_additional_layer' if layer_addition else ''}{'_removed_layer' if layer_removal else ''}.pth"
        model_name = f"My CNN model with {num_epochs} epochs and kernel size {kernel_size}{' and removed layer' if layer_removal else ' and additional layer' if layer_addition else ''}"
        # Load the CIFAR-10 dataset only if not already loaded
        if self.cnn_train_loader is None or self.cnn_test_loader is None:
            print("Loading CIFAR-10 dataset for CNN...")
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.cnn_train_loader, self.cnn_test_loader = load_cifar10(transform, batch_size=32)

        cnn_model = ConvolutionNeuralNetwork(device=self.device, num_epochs=num_epochs, kernel_size=kernel_size,
                                         layer_removal=layer_removal, layer_addition=layer_addition, file_path=file_path)
        test_pred = cnn_model.fit(dataset=self.cnn_train_loader.dataset).predict(data_loader=self.cnn_test_loader)
        self.append_report(evaluate_model(model_name, self.test_labels, test_pred, self.classes, self.FIGURE_PATH))
