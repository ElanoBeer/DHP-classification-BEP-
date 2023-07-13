# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import re

# Load TQDM to Show Progress Bars
from tqdm import tqdm

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

# hand-crafted functions
from pre_processing import *
from utilities import average_embeddings, random_shuffle
from model_settings import *

# warning avoidance
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


# --------------------------------------------------------------------------------------------------------
# Implement uncertainty sampling in training
# --------------------------------------------------------------------------------------------------------


def uncertainty_batch_teaching(
    w2v_models: dict,
    models: list,
    X: pd.Series,
    y: pd.Series,
    max_samples: int,
    batch_size=20,
    init_size=20,
    method="f1",
    smoothed=False,
):
    """A function that implements uncertainty sampling during training through batch teaching.
    It furthermore visualizes the active learning procedure through line plots of a metric per model.

    Args:
        w2v_models (dict): A dictionary that contains the w2v_model to use per classification model.
        models (list): A list that contains the classification models.
        X (pd.Series): The pre-processed training dataset.
        y (pd.Series): The labels.
        max_samples (int): The maximum number of samples to be trained.
        batch_size (int, optional): The size of the batches that are queried. Defaults to 20.
        init_size (int, optional): The size of the initial training set. Defaults to 20.
        method (str, optional): The evaluation metric to be visualized. Defaults to "f1".
        smooted (boolean): A boolean representation to smooth the plot. Defauls to False.

    Returns:
        acc_lst (list): A list with accuracy scores per model.
        f1_lst (list): A list with f1 scores per model.
    """

    acc_lst = []  # list: stores the accuracy scores per model
    f1_lst = []  # list: stores the f1 scores per model

    for model in tqdm(models):
        w2v_model = w2v_models[model]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=5
        )

        # Average word embeddings
        X_train_vect_avg, X_test_vect_avg = average_embeddings(
            w2v_model, X_train, X_test
        )

        # Initial labeled samples
        # Randomly oversample as SMOTE would not have enough neighbours
        X_initial, y_initial = oversampler.fit_resample(
            X_train_vect_avg[:init_size], y_train[:init_size]
        )
        X_initial, y_initial = random_shuffle(X_initial, y_initial)

        # Class balancing with SMOTE
        X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)
        X_train, y_train, X_test, y_test = map(
            np.array, (X_over, y_over, X_test_vect_avg, y_test)
        )

        # Query strategy -> margin sampling reduces to uncertainty sampling
        # for this LinearSVC implementation
        query_strategy = uncertainty_sampling if model != lsvc else margin_sampling

        # Create active learner
        learner = ActiveLearner(
            estimator=model,
            query_strategy=query_strategy,
            X_training=list(X_initial),
            y_training=list(y_initial),
        )

        model_acc_lst = []
        model_f1_lst = []

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

            current_batch_size = init_size
            while current_batch_size <= max_samples:
                if model != lsvc:
                    query_idx, _ = learner.query(
                        X_train, n_instances=current_batch_size
                    )
                else:
                    # Calculate the distances from the decision boundary
                    decision_values = model.decision_function(X_train)
                    distances = np.abs(decision_values) / np.linalg.norm(model.coef_)
                    query_idx = np.argsort(distances)[:current_batch_size]

                # teach only the new entries
                learner.teach(X_train[query_idx], y_train[query_idx], only_new=True)
                y_pred = learner.predict(X_test_vect_avg)
                model_acc_lst.append(accuracy_score(y_test, y_pred))
                model_f1_lst.append(f1_score(y_test, y_pred))

                current_batch_size += batch_size

            # Append model performances seperately
            acc_lst.append(model_acc_lst)
            f1_lst.append(model_f1_lst)

    # Plot method per number of samples
    for i, model in enumerate(models):
        model_name = re.search(r"\w+(?=\()", str(model)).group()
        scores = np.array(acc_lst[i]) if method == "accuracy" else np.array(f1_lst[i])
        x = np.array(list(range(batch_size, max_samples + 1, batch_size)))

        # If-statement that splits between a smoothed visualization or not.
        if smoothed:
            coeffs = np.polyfit(x, scores, 10)  # Fit a polynomial regression curve
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 300)  # Generate a smoother x-axis
            y_smooth = poly(x_smooth)
            plt.plot(x_smooth, y_smooth, label=model_name)
        else:
            plt.plot(x, scores, label=model_name)

    if method == "accuracy":
        plt.title("Plot of Accuracy per Number of Samples")
        plt.ylabel("Accuracy")
    else:
        plt.title("Plot of F1 Score per Number of Samples")
        plt.ylabel("F1 Score")

    plt.xlabel("Number of samples")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    return acc_lst, f1_lst


# Example usage
acc_lst, f1_lst = uncertainty_batch_teaching(
    w2v_models=w2v_models,
    models=[lgcv, rfc, lsvc],
    X=labeled_lemma_df["text_clean"],
    y=labeled_lemma_df["label"],
    max_samples=len(labeled_lemma_df),
    smoothed=True,
)
