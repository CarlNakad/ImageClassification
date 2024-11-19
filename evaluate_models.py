from sklearn.metrics import confusion_matrix, classification_report


def evaluate_model(model_name, test_true, test_pred, classes):
    test_labels_named = [classes[label] for label in test_true]
    # test_pred_named = [classes[label] for label in test_pred]
    print(f"Evaluating model: {model_name}")
    # Confusion Matrix
    cm = confusion_matrix(test_true, test_pred)
    print(f"Confusion Matrix: \n{cm}")

    # Classification Report for accuracy, precision, recall, and F1-measure.
    report = classification_report(test_true, test_pred)
    print(f"Classification Report: \n{report}")
