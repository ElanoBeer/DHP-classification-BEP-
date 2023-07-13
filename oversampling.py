# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load TQDM to Show Progress Bars #
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

# spacy stopwords
from spacy.lang.en import STOP_WORDS

STOP_WORDS = list(STOP_WORDS)

# sklearn
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

# imblearn
from imblearn.over_sampling import *

import gensim

# hand-crafted functions
from pre_processing import *
from utilities import average_embeddings

# ---------------------------------------------------------------------------------------------------------
# Create the pre-processed dataset
# ---------------------------------------------------------------------------------------------------------

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
labels = pd.read_excel("../Datasets/final_sample_Elano.xlsx")

text_df = preprocess(df, labels, "baseline")
labeled_df, unlabeled_df = label_split(text_df)

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

# ---------------------------------------------------------------------------------------------------------
# Oversampling grouped barchart
# ---------------------------------------------------------------------------------------------------------

# Initialize a dictionary to store the evaluation metrics and predictions for each run and each classifier
results = {"lgcv": [], "rfc": [], "lsvc": []}

# Define the oversampling methods
oversampling_methods = {
    "RandomOverSampler": RandomOverSampler(
        sampling_strategy="minority", random_state=5
    ),
    "SMOTE": SMOTE(sampling_strategy="minority", random_state=5),
    "ADASYN": ADASYN(sampling_strategy="minority", random_state=5),
    "BorderlineSMOTE": BorderlineSMOTE(sampling_strategy="minority", random_state=5),
    "SVMSMOTE": SVMSMOTE(sampling_strategy="minority", random_state=5),
}

# Run the simulation 50 times
for _ in tqdm(range(50)):
    X_train, X_test, y_train, y_test = train_test_split(
        labeled_df["text_clean"], labeled_df["label"], test_size=0.3, random_state=5
    )
    X_train, X_test = average_embeddings(w2v_model, X_train, X_test)

    for clf, clf_name in zip((lgcv, rfc, lsvc), ("lgcv", "rfc", "lsvc")):
        for method_name, oversampler in oversampling_methods.items():
            X_over, y_over = oversampler.fit_resample(X_train, y_train)
            clf.fit(X_over, y_over)
            train_predictions = clf.predict(X_over)
            test_predictions = clf.predict(X_test)
            train_f1score = f1_score(y_over, train_predictions)
            test_f1score = f1_score(y_test, test_predictions)
            results[clf_name].append(
                {
                    "oversampling_method": method_name,
                    "train_f1score": train_f1score,
                    "test_f1score": test_f1score,
                }
            )

# Define the oversampling techniques
techniques = ["RandomOverSampler", "SMOTE", "ADASYN", "BorderlineSMOTE", "SVMSMOTE"]


lgcv_f1scores = []
rfc_f1scores = []
lsvc_f1scores = []

for method_name in oversampling_methods:
    lgcv_scores = [
        result["test_f1score"]
        for result in results["lgcv"]
        if result["oversampling_method"] == method_name
    ]
    rfc_scores = [
        result["test_f1score"]
        for result in results["rfc"]
        if result["oversampling_method"] == method_name
    ]
    lsvc_scores = [
        result["test_f1score"]
        for result in results["lsvc"]
        if result["oversampling_method"] == method_name
    ]

    lgcv_avg_f1score = sum(lgcv_scores) / len(lgcv_scores)
    rfc_avg_f1score = sum(rfc_scores) / len(rfc_scores)
    lsvc_avg_f1score = sum(lsvc_scores) / len(lsvc_scores)

    lgcv_f1scores.append(lgcv_avg_f1score)
    rfc_f1scores.append(rfc_avg_f1score)
    lsvc_f1scores.append(lsvc_avg_f1score)


# Set the width of the bars
bar_width = 0.2

# Set the x-axis positions for the bars
index = [1, 2, 3, 4, 5]

# Set the color palette
sns.set_palette("viridis")

# Plot the average F1 scores
plt.figure(figsize=(10, 6))
plt.bar(index, lgcv_f1scores, width=bar_width, label="LogisticRegression")
plt.bar(
    [i + bar_width for i in index],
    rfc_f1scores,
    width=bar_width,
    label="RandomForestClassifier",
)
plt.bar(
    [i + 2 * bar_width for i in index],
    lsvc_f1scores,
    width=bar_width,
    label="LinearSVC",
)
plt.xlabel("Oversampling Techniques")
plt.ylabel("F1score")
plt.title("Performance Comparison - F1score")
plt.xticks([i + bar_width for i in index], techniques)
plt.legend()
plt.show()
