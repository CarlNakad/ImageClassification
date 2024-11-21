import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model_name, test_true, test_pred, classes, output_path):
    """
    Evaluate the model using confusion matrix and classification report to get accuracy, precision, recall, and F1-measure.

    :param model_name: Model name
    :param test_true: True labels
    :param test_pred: Predicted labels
    :param classes: Class names
    :return: Dictionary containing model name, accuracy, precision, recall, and F1-measure
    """
    # Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)

    # Plot Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('bottom')
    plt.xticks(rotation=45, ha='left')

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    plt.savefig(f"{output_path}/{model_name} confusion matrix.png")

    # Classification Report for accuracy, precision, recall, and F1-measure.
    report = classification_report(test_true, test_pred, target_names=classes, output_dict=True, zero_division=0)

    # Extract key metrics (average over all classes)
    avg_metrics = report['weighted avg']
    return {
        "Model": model_name,
        "Accuracy": report['accuracy'],
        "Precision": avg_metrics['precision'],
        "Recall": avg_metrics['recall'],
        "F1-Measure": avg_metrics['f1-score']
    }


def generate_comparison_table(models_report, output_path):
    """
    Generate a comparison table for model performance.
    x axis: All evaluated models
    y axis: Metrics (Accuracy, Precision, Recall, F1-Measure)

    :param models_report:
    :return:
    """
    metrics_df = pd.DataFrame(models_report)

    # Plot as a heatmap with switched x and y axes
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df.set_index('Model'), annot=True, fmt=".2f", cmap='YlGnBu', cbar=True)

    plt.title('Model Performance Comparison')
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.ylabel('')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_path}/model_performance_comparison.png")
    print(f"Model performance comparison table saved at {output_path}/model_performance_comparison.png")
    return metrics_df