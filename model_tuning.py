# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load TQDM to Show Progress Bars #
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    roc_auc_score,
    roc_curve,
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

# warning avoidance
import warnings
from sklearn.exceptions import ConvergenceWarning

# ---------------------------------------------------------------------------------------------------------
# Create the pre-processed datasets
# ---------------------------------------------------------------------------------------------------------

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
labels = pd.read_excel("../Datasets/final_sample_Elano.xlsx")

lemma_df = preprocess(df, labels, "lemmatization")
labeled_lemma_df, unlabeled_lemma_df = label_split(lemma_df)

# --------------------------------------------------------------------------------------------------------
# Decide the model settings
# --------------------------------------------------------------------------------------------------------

lgr_parameters = {
    "Cs": [5, 10, 20],
    "cv": [5, 10],
    "max_iter": [500, 2000],
    "solver": ["lbfgs", "liblinear"],
    "scoring": ["f1"],
    "class_weight": ["balanced"],
}
rf_parameters = {
    "n_estimators": [50, 200, 500],
    "max_depth": [5, 10, None],
    "max_features": ["sqrt", "log2"],
    "class_weight": ["balanced"],
}
svc_parameters = {
    "penalty": ["l1", "l2"],
    "tol": [0.5, 0.001],
    "C": [1, 5],
    "class_weight": [None, "balanced"],
}

lgcv = GridSearchCV(LogisticRegressionCV(), lgr_parameters, scoring="f1", refit="f1")
rfc = GridSearchCV(RandomForestClassifier(), rf_parameters, scoring="f1", refit="f1")
lsvc = GridSearchCV(LinearSVC(), svc_parameters, scoring="f1", refit="f1")

w2v_model_1 = gensim.models.Word2Vec(
    lemma_df["text_clean"], vector_size=100, window=2, min_count=10, sg=1
)

w2v_model_2 = gensim.models.Word2Vec(
    lemma_df["text_clean"], vector_size=100, window=10, min_count=5, sg=1
)

smote = SMOTE(sampling_strategy="minority", random_state=5)


# --------------------------------------------------------------------------------------------------------
# Find the best parameter combination for the three classification models
# --------------------------------------------------------------------------------------------------------

# Initialize a dictionary to store the evaluation metrics and predictions for each run and each classifier
results = {"lgcv": [], "rfc": [], "lsvc": []}

X_train, X_test, y_train, y_test = train_test_split(
    labeled_lemma_df["text_clean"],
    labeled_lemma_df["label"],
    test_size=0.3,
    random_state=5,
)
X_train_vect_avg_1, X_test_vect_avg_1 = average_embeddings(w2v_model_1, X_train, X_test)
X_train_vect_avg_2, X_test_vect_avg_2 = average_embeddings(w2v_model_2, X_train, X_test)

# Ignore convergence warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    lgcv.fit(X_train_vect_avg_1, y_train)
    rfc.fit(X_train_vect_avg_2, y_train)
    lsvc.fit(X_train_vect_avg_2, y_train)

    best_params_lgcv = lgcv.best_params_
    best_params_rfc = rfc.best_params_
    best_params_lsvc = lsvc.best_params_

print("Best Parameters for Logistic Regression:")
print(best_params_lgcv)


print("Best Parameters for Random Forest Classifier:")
print(best_params_rfc)


print("Best Parameters for Linear SVC:")
print(best_params_lsvc)

tuned_lgcv = LogisticRegressionCV(**best_params_lgcv)
tuned_rfc = RandomForestClassifier(**best_params_rfc)
tuned_lsvc = LinearSVC(**best_params_lsvc)

# Perform the simulation 50 times
for i in tqdm(range(50)):
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_lemma_df["text_clean"],
        labeled_lemma_df["label"],
        test_size=0.3,
        random_state=5,
    )

    # Model-specific word embedding and tuned model assignments
    if i % 3 == 0:  # lgcv uses w2v_model_1 and tuned_lgcv
        w2v_model = w2v_model_1
        tuned_model = tuned_lgcv
        clf_name = "lgcv"
    else:  # rfc and lsvc use w2v_model_2 and their respective tuned models
        w2v_model = w2v_model_2
        if i % 3 == 1:  # rfc uses tuned_rfc
            tuned_model = tuned_rfc
            clf_name = "rfc"
        else:  # lsvc uses tuned_lsvc
            tuned_model = tuned_lsvc
            clf_name = "lsvc"

    X_train_vect_avg, X_test_vect_avg = average_embeddings(w2v_model, X_train, X_test)

    # Ignore convergence warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        tuned_model.fit(X_train_vect_avg, y_train)
        test_predictions = tuned_model.predict(X_test_vect_avg)
        metrics_test = {
            "accuracy": accuracy_score(y_test, test_predictions),
            "recall": recall_score(y_test, test_predictions),
            "precision": precision_score(y_test, test_predictions),
            "fnr": fnr(y_test, test_predictions),
            "f1score": f1_score(y_test, test_predictions),
        }
        results[clf_name].append(
            {"test_predictions": test_predictions, "test_metrics": metrics_test}
        )


# --------------------------------------------------------------------------------------------------------
# Create a Precision-Recall curve and AUC-ROC curve for the three models
# --------------------------------------------------------------------------------------------------------

# Generate baseline prediction (majority class)
ns_probs = [0 for _ in range(len(y_test))]

# Create a list of models and their corresponding names
models = [
    (lgcv, "LogisticRegression", w2v_model_1),
    (tuned_rfc, "RandomForestClassifier", w2v_model_2),
    (tuned_lsvc, "LinearSVC", w2v_model_2),
]

# Create subplots for ROC and PR curves
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plot the PR curve and ROC curve for each model
for model, model_name, w2v_model in models:
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_lemma_df["text_clean"],
        labeled_lemma_df["label"],
        test_size=0.3,
        random_state=5,
    )
    X_train_vect_avg, X_test_vect_avg = average_embeddings(w2v_model, X_train, X_test)
    X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)
    X_train, y_train, X_test, y_test = map(
        np.array, (X_over, y_over, X_test_vect_avg, y_test)
    )

    if model_name == "LinearSVC":
        # Use decision function and sigmoid to obtain probabilities
        model_scores = model.decision_function(X_test)
        model_probs = 1 / (1 + np.exp(-model_scores))
    else:
        # Predict probabilities
        model_probs = model.predict_proba(X_test)[:, 1]

    # Calculate AUC score
    model_auc = roc_auc_score(y_test, model_probs)

    # Calculate AP score
    avg_precision = average_precision_score(y_test, model_probs)

    # Calculate ROC curve
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

    # Calculate PR curve
    model_precision, model_recall, _ = precision_recall_curve(y_test, model_probs)

    # Plot the PR curve
    display = PrecisionRecallDisplay.from_predictions(
        y_test, model_probs, name=model_name, ax=ax1
    )

    # Plot the ROC curve
    ax2.plot(
        model_fpr, model_tpr, marker=".", label=f"{model_name} (AUC: {model_auc:.2f})"
    )

ns_ap = average_precision_score(y_test, ns_probs)
display = PrecisionRecallDisplay.from_predictions(
    y_test, ns_probs, name="Baseline", ax=ax1
)
display.ax_.set_title("Precision-Recall Curve (PR)")

# Plot the baseline (majority class) ROC curve
ns_auc = roc_auc_score(y_test, ns_probs)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
ax2.plot(ns_fpr, ns_tpr, linestyle="--", label="Baseline (AUC: 0.50)")

# Axis labels and title for PR curve
ax1.set_xlabel("Recall")
ax1.set_ylabel("Precision")
ax1.set_title("Precision-Recall Curve (PR)")
ax1.legend()
ax1.grid(True)

# Axis labels and title for ROC curve
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("Receiver Operating Characteristic Curve (ROC)")
ax2.legend()
ax2.grid(True)

# Adjust spacing between subplots
fig.tight_layout()

# Show the combined visualization
plt.show()
