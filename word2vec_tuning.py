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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
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

# futher utilities
import itertools
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
# Decide the baseline model settings
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
    lemma_df["text_clean"], vector_size=100, window=5, min_count=2
)

smote = SMOTE(sampling_strategy="minority", random_state=5)

# --------------------------------------------------------------------------------------------------------
# Create a heatmap that shows f1-scores for word2vec manual gridsearch hyperparameter tuning
# --------------------------------------------------------------------------------------------------------

# Initialize a list to store the evaluation metrics and predictions for each run and each classifier
results = {"lgcv": [], "rfc": [], "lsvc": []}

# Define the parameter values to simulate
min_counts = [2, 5, 10]
vector_size = 100
windows = [2, 5, 10]
sg = [0, 1]
corpus = lemma_df["text_clean"]

# Generate all combinations of the parameters
parameters = list(itertools.product(min_counts, windows, sg))

for params in tqdm(parameters):
    min_count, window, sg = params

    # Create the Word2Vec model with the specified parameters
    w2v_model = gensim.models.Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
    )

    # perform the simulation 10 times
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(
            labeled_lemma_df["text_clean"],
            labeled_lemma_df["label"],
            test_size=0.3,
            random_state=5,
        )
        X_train_vect_avg, X_test_vect_avg = average_embeddings(
            w2v_model, X_train, X_test
        )
        X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)

        X_train, y_train, X_test, y_test = map(
            np.array, (X_over, y_over, X_test_vect_avg, y_test)
        )

        # Evaluate the classifiers and store the evaluation metrics and predictions
        for clf, clf_name in zip((lgcv, rfc, lsvc), ("lgcv", "rfc", "lsvc")):
            # Compute the evaluation metrics and store the predictions and metrics for each classifier on the test set
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                clf.fit(X_train, y_train)
                test_predictions = clf.predict(X_test)
                results[clf_name].append(f1_score(y_test, test_predictions))


best_params = {"lgcv": None, "rfc": None, "lsvc": None}
avg_results_dict = []

for clf_name, clf_results in results.items():
    num_sets = len(clf_results) // 10  # based on complete sets of 10 values

    best_avg_results = float("-inf")
    best_params_index = None

    for i in range(num_sets):
        start_index = i * 10
        end_index = start_index + 10

        # Extract the set of results for the current parameter combination
        set_results = clf_results[start_index:end_index]

        # Calculate the average of the evaluation metric results for the current set
        avg_results = np.mean(set_results)
        avg_results_dict.append((clf_name, avg_results))

        if avg_results > best_avg_results:
            best_avg_results = avg_results
            best_params_index = i

    # Retrieve the parameters corresponding to the best set of results
    best_params[clf_name] = parameters[best_params_index]
    print(best_avg_results)

for clf_name, params in best_params.items():
    print(f"Best parameters for {clf_name}: {params}")


lgcv_results = [item[1] for item in avg_results_dict[0:18]]
rfc_results = [item[1] for item in avg_results_dict[18:36]]
lsvc_results = [item[1] for item in avg_results_dict[36:]]

heatmap_df = pd.DataFrame(
    {
        "LogisticRegression": lgcv_results,
        "RandomForestClassifier": rfc_results,
        "LinearSVC": lsvc_results,
    },
    index=parameters,
)

fig, ax = plt.subplots()
sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="viridis")
# Set the x-axis label on top of the heatmap
ax.xaxis.set_label_position("top")

# Set the axis labels and tick labels
ax.set_yticklabels(parameters, rotation=0)
ax.set_xticklabels(heatmap_df.columns, rotation=45)

# Set the plot title and labels
plt.title("Word2Vec Tuning Heatmap")
plt.xlabel("Classification Model")
plt.ylabel("Parameter Combination")
