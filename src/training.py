from transformers import Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred):
    """Compute classification metrics for Transformer-based model predictions.

    This function is designed to be passed to a HuggingFace `Trainer` in order
    to evaluate model performance after each epoch. It extracts the predicted
    class labels and true labels, computes precision, recall, F1-score, and
    accuracy, and returns them in a dictionary.

    Args:
        pred: An object of type `EvalPrediction` containing:
            - `predictions`: model output logits of shape (N, num_classes)
            - `label_ids`: ground-truth class indices of shape (N,)

    Returns:
        dict: A dictionary containing the following metrics:
            - 'accuracy' (float): overall prediction accuracy.
            - 'f1' (array or float): F1-score per class (or averaged depending on config).
            - 'precision' (array or float): precision per class.
            - 'recall' (array or float): recall per class.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
