# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC

# Imblearn
from imblearn.over_sampling import SMOTE, RandomOverSampler

import gensim

# hand-crafted functions
from pre_processing import *
from utilities import average_embeddings, random_shuffle

# active learning modAL
# pip install modAL-python
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling, margin_sampling

# ---------------------------------------------------------------------------------------------------------
# Create the pre-processed datasets
# ---------------------------------------------------------------------------------------------------------

df = pd.read_excel("../Datasets/HealthPlatforms_BEP.xlsx")
labels = pd.read_excel("../Datasets/final_sample_Elano.xlsx")

lemma_df = preprocess(df, labels, "lemmatization")
labeled_lemma_df, unlabeled_lemma_df = label_split(lemma_df)

# --------------------------------------------------------------------------------------------------------
# Tuned model settings
# --------------------------------------------------------------------------------------------------------

lgcv = LogisticRegressionCV(
    Cs=10, cv=5, max_iter=500, solver="lbfgs", scoring="f1", class_weight="balanced"
)
rfc = RandomForestClassifier(
    n_estimators=200, max_depth=10, max_features="log2", class_weight="balanced"
)
lsvc = LinearSVC(C=5, penalty="l2", tol=0.0001, class_weight=None)

w2v_model_1 = gensim.models.Word2Vec(
    lemma_df["text_clean"], vector_size=100, window=2, min_count=10, sg=1
)

w2v_model_2 = gensim.models.Word2Vec(
    lemma_df["text_clean"], vector_size=100, window=10, min_count=5, sg=1
)

smote = SMOTE(sampling_strategy="minority", random_state=5)
oversampler = RandomOverSampler(sampling_strategy="minority", random_state=5)

# Word2Vec models
w2v_models = {lgcv: w2v_model_1, rfc: w2v_model_2, lsvc: w2v_model_2}
