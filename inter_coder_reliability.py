import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

# import the independent labels of the random experiment sample
coder_1 = pd.read_excel("../Datasets/elano_experiment_sample.xlsx")
coder_2 = pd.read_excel("../Datasets/marco_experiment_sample.xlsx")

# rename the label columns
coder_1.rename(columns={"social_platform": "social_platform_1"}, inplace=True)
coder_2.rename(columns={"social": "social_platform_2"}, inplace=True)

# create a dataframe with both labels
coder_df = pd.DataFrame(
    data=coder_2[["id", "new_description", "social_platform_2"]]
).join(coder_1[["social_platform_1"]])


# ---------------------------------------------------------------------------------
# Intercoder reliability analysis
# ---------------------------------------------------------------------------------

# plot the confusion matrix
coder_cm = confusion_matrix(
    coder_df["social_platform_1"], coder_df["social_platform_2"]
)
ax = sns.heatmap(coder_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
ax.set(xlabel="Predicted label", ylabel="True label")

# create a dataframe with classification metrics
classification = classification_report(
    coder_df["social_platform_1"], coder_df["social_platform_2"], output_dict=True
)
icr_df = pd.DataFrame(classification).transpose()[["precision", "recall", "f1-score"]]

cohen_kappa_score(coder_df["social_platform_1"], coder_df["social_platform_2"])


# Obtain instances of disagreement
disagreement_df = pd.DataFrame(
    coder_df.loc[
        coder_df["social_platform_1"] != coder_df["social_platform_2"],
        "new_description",
    ]
)
