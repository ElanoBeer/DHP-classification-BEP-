# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load TQDM to Show Progress Bars #
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Spacy stopwords
from spacy.lang.en import STOP_WORDS

STOP_WORDS = list(STOP_WORDS)

# Sklearn
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

# Imblearn
from imblearn.over_sampling import SMOTE

import gensim

# hand-crafted functions
from pre_processing import *
from utilities import average_embeddings, fnr


# ---------------------------------------------------------------------------------------------------------
# Create the pre-processed datasets
# ---------------------------------------------------------------------------------------------------------

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
labels = pd.read_excel("../Datasets/final_sample_Elano.xlsx")

text_df = preprocess(df, labels, "baseline")
labeled_df, unlabeled_df = label_split(text_df)

stem_df = preprocess(df, labels, "stemming")
labeled_stem_df, unlabeled_stem_df = label_split(stem_df)

lemma_df = preprocess(df, labels, "lemmatization")
labeled_lemma_df, unlabeled_lemma_df = label_split(lemma_df)

# --------------------------------------------------------------------------------------------------------
# Decide the model settings
# --------------------------------------------------------------------------------------------------------

lgcv = LogisticRegressionCV(
    scoring="f1", solver="lbfgs", max_iter=2000, class_weight="balanced"
)
rfc = RandomForestClassifier(max_depth=10, random_state=5, class_weight="balanced")
lsvc = LinearSVC(
    penalty="l2",
    tol=0.5,
    C=5,
    multi_class="ovr",
    fit_intercept=True,
    intercept_scaling=1,
    class_weight="balanced",
)

w2v_model = gensim.models.Word2Vec(
    text_df["text_clean"], vector_size=100, window=5, min_count=2
)

smote = SMOTE(sampling_strategy="minority", random_state=5)

# --------------------------------------------------------------------------------------------------------
# Create a simulation function
# --------------------------------------------------------------------------------------------------------


def simulation(labeled_df: pd.DataFrame, cv=5, n_iter=10):
    """A function that runs a simulation where cross-validation
    is used to train and evaluate the pre-defined models.
    It includes classical metrics with the addition of the fnr,
    ultimately returning multiple results.

    Args:
        labeled_df (pd.DataFrame): The preprocessed dataframe with labels.
        cv (int, optional): The number of k's during cross-validation. Defaults to 5.

    Returns:
        cv_results (dict): The dictionary that contains the cross-validation scores.
        results (dict): The dictionary that contains the predictions
                        and classification metrics for each model using smote.
        y_test (array): At the end, the last y_test is returned.
    """

    # Initialize a list to store the evaluation metrics and predictions for each run and each classifier
    results = {"lgcv": [], "rfc": [], "lsvc": []}

    # Run the simulation n_iter times with 5-fold cross-validation and a test set
    for _ in tqdm(range(n_iter)):
        # Define the cross-validation strategy
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            labeled_df["text_clean"], labeled_df["label"], test_size=0.3, random_state=5
        )
        X_train_vect_avg, X_test_vect_avg = average_embeddings(
            w2v_model, X_train, X_test
        )
        X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)

        X_train, y_train, X_test, y_test = map(
            np.array, (X_over, y_over, X_test_vect_avg, y_test)
        )

        # Evaluate the classifiers using cross-validation and store the evaluation metrics and predictions
        for clf, clf_name in zip((lgcv, rfc, lsvc), ("lgcv", "rfc", "lsvc")):
            scoring = ["accuracy", "recall", "precision", "f1"]
            cv_results = cross_validate(
                clf, X_train, y_train, cv=cv, scoring=scoring, return_estimator=True
            )

            # Compute the evaluation metrics and store the predictions and metrics for each classifier on the test set
            for estimator in cv_results["estimator"]:
                train_predictions = estimator.predict(X_train)
                test_predictions = estimator.predict(X_test)
                metrics_train = {
                    "accuracy": accuracy_score(y_train, train_predictions),
                    "recall": recall_score(y_train, train_predictions),
                    "precision": precision_score(y_train, train_predictions),
                    "fnr": fnr(y_train, train_predictions),
                    "f1score": f1_score(y_train, train_predictions),
                }
                metrics_test = {
                    "accuracy": accuracy_score(y_test, test_predictions),
                    "recall": recall_score(y_test, test_predictions),
                    "precision": precision_score(y_test, test_predictions),
                    "fnr": fnr(y_test, test_predictions),
                    "f1score": f1_score(y_test, test_predictions),
                }
                results[clf_name].append(
                    {
                        "train_predictions": train_predictions,
                        "test_predictions": test_predictions,
                        "train_metrics": metrics_train,
                        "test_metrics": metrics_test,
                    }
                )

    return cv_results, results, y_test


# --------------------------------------------------------------------------------------------------------
# Obtain the results for different pre-processed datasets
# --------------------------------------------------------------------------------------------------------

cv_results, results, last_ytest = simulation(labeled_df)
cv_stem_results, stem_results, last_stem_ytest = simulation(labeled_stem_df)
cv_lemma_results, lemma_results, last_lemma_ytest = simulation(labeled_lemma_df)

metrics = ["accuracy", "recall", "precision", "fnr", "f1score"]
results_df = pd.DataFrame(
    columns=["Metric", "Minimum", "Maximum", "Average", "Classifier"]
)

for clf_name in ("lgcv", "rfc", "lsvc"):
    all_train_metrics = [
        r["train_metrics"]
        for r in results[clf_name]
        if r["train_predictions"] is not None
    ]
    all_test_metrics = [
        r["test_metrics"]
        for r in results[clf_name]
        if r["test_predictions"] is not None
    ]

    min_train_metrics = [
        np.min([m[metric] for m in all_train_metrics]) for metric in metrics
    ]
    max_train_metrics = [
        np.max([m[metric] for m in all_train_metrics]) for metric in metrics
    ]
    avg_train_metrics = [
        np.mean([m[metric] for m in all_train_metrics]) for metric in metrics
    ]
    min_test_metrics = [
        np.min([m[metric] for m in all_test_metrics]) for metric in metrics
    ]
    max_test_metrics = [
        np.max([m[metric] for m in all_test_metrics]) for metric in metrics
    ]
    avg_test_metrics = [
        np.mean([m[metric] for m in all_test_metrics]) for metric in metrics
    ]

    for i, metric in enumerate(metrics):
        results_df = results_df.append(
            {
                "Metric": metric,
                "Minimum": min_train_metrics[i],
                "Maximum": max_train_metrics[i],
                "Average": avg_train_metrics[i],
                "Classifier": clf_name + " (Train)",
            },
            ignore_index=True,
        )
        results_df = results_df.append(
            {
                "Metric": metric,
                "Minimum": min_test_metrics[i],
                "Maximum": max_test_metrics[i],
                "Average": avg_test_metrics[i],
                "Classifier": clf_name + " (Test)",
            },
            ignore_index=True,
        )

results_df

# --------------------------------------------------------------------------------------------------------
# Create boxplots per metrics and normalization step as well as one confusion matrix prediction
# --------------------------------------------------------------------------------------------------------

# Compute the evaluation metrics over all runs for each classifier on the test set
metric_names = ["accuracy", "recall", "precision", "fnr", "f1score"]
metric_values = {
    clf_name: {
        metric: [
            r["test_metrics"][metric]
            for r in lemma_results[clf_name]
            if r["test_predictions"] is not None
        ]
        for metric in metric_names
    }
    for clf_name in ("lgcv", "rfc", "lsvc")
}

# Plot boxplots for each evaluation metric in a horizontal line
fig, axes = plt.subplots(nrows=1, ncols=len(metric_names), figsize=(16, 6))

for i, metric in enumerate(metric_names):
    data = [metric_values[clf_name][metric] for clf_name in ("lgcv", "rfc", "lsvc")]
    ax = axes[i]
    ax.set_title(f"Boxplot of {metric}")
    sns.boxplot(data=data, orient="v", ax=ax)
    ax.set_ylabel(metric)

plt.tight_layout()
plt.show()


# Get the last prediction made by each classifier on the test set
lgcv_last_prediction = results["lgcv"][-1]["test_predictions"]
rfc_last_prediction = results["rfc"][-1]["test_predictions"]
lsvc_last_prediction = results["lsvc"][-1]["test_predictions"]

# Create the confusion matrix for each classifier
lgcv_cm = confusion_matrix(last_lemma_ytest, lgcv_last_prediction)
rfc_cm = confusion_matrix(last_lemma_ytest, rfc_last_prediction)
lsvc_cm = confusion_matrix(last_lemma_ytest, lsvc_last_prediction)

# Define the class labels (replace with your actual class labels if needed)
class_labels = [0, 1]

# Plot the confusion matrix for each classifier
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

for i, (cm, clf_name) in enumerate(
    zip([lgcv_cm, rfc_cm, lsvc_cm], ["lgcv", "rfc", "lsvc"])
):
    ax = axes[i]
    cmn = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cmn, annot=cm, fmt=".2f", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {clf_name}")
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels, rotation=0)

plt.tight_layout()
plt.show()
