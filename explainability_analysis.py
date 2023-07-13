# ---------------------------------------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------------------------------------

from model_settings import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# warning avoidance
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning

# ---------------------------------------------------------------------------------------------------------
# Fit the best model
# ---------------------------------------------------------------------------------------------------------

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    labeled_lemma_df["text_clean"],
    labeled_lemma_df["label"],
    test_size=0.3,
    random_state=5,
)

# Average word embeddings
X_train_vect_avg, X_test_vect_avg = average_embeddings(w2v_model_2, X_train, X_test)

# Class balancing with SMOTE
X_over, y_over = smote.fit_resample(X_train_vect_avg, y_train)
X_train, y_train, X_test, y_test = map(
    np.array, (X_over, y_over, X_test_vect_avg, y_test)
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    lgcv.fit(X_train, y_train)
print(classification_report(y_test, lgcv.predict(X_test)))

# ---------------------------------------------------------------------------------------------------------
# Most similar words experiment
# ---------------------------------------------------------------------------------------------------------

important_words = [
    "network",
    "platform",
    "social",
    "community",
    "review",
    "healthcare",
    "patient",
    "doctor",
    "solution",
    "connect",
]

[w2v_model_1.wv.most_similar(word)[0:5] for word in important_words]

# ---------------------------------------------------------------------------------------------------------
# Word feature importance analysis
# ---------------------------------------------------------------------------------------------------------


def word_feature_plot(w2v_model, model):
    """_summary_

    Args:
        w2v_model (_type_): _description_
        model (_type_): _description_
    """
    # Obtain the vocabulary
    word2vec_vocabulary = list(w2v_model.wv.key_to_index.keys())

    if model == rfc:
        # Obtain feature importances
        feature_importances = rfc.feature_importances_

        # Create a list of (word, coefficient) tuples
        word_coefficient_pairs = [
            (word, coefficient)
            for word, coefficient in zip(word2vec_vocabulary, feature_importances)
        ]

        # Sort the word_coefficient_pairs based on the coefficient value
        sorted_pairs = sorted(word_coefficient_pairs, key=lambda x: x[1])

        # Get the top 10 most negative words
        most_negative_words = sorted_pairs[:10]

        # Get the top 10 most positive words
        most_positive_words = sorted_pairs[-10:][::-1]
        print(most_negative_words)

    else:
        # Get the coefficients of the logistic regression model
        coefficients = model.coef_[0]

        # Create a list of (word, coefficient) tuples
        word_coefficient_pairs = [
            (word, coefficient)
            for word, coefficient in zip(word2vec_vocabulary, coefficients)
        ]

        # Sort the word_coefficient_pairs based on the coefficient value
        sorted_pairs = sorted(word_coefficient_pairs, key=lambda x: x[1])

        # Get the top 10 most negative words
        most_negative_words = sorted_pairs[:10]

        # Get the top 10 most positive words
        most_positive_words = sorted_pairs[-10:][
            ::-1
        ]  # Reverse the order to get the most positive words

    # Create the horizontal barchart subplot
    negative_df = (
        pd.DataFrame(most_negative_words).sort_values(1, ascending=False).tail(10)
    )
    positive_df = (
        pd.DataFrame(most_positive_words).sort_values(1, ascending=True).tail(10)
    )

    negative_df.rename(columns={0: "words", 1: "Coefficients"}, inplace=True)
    positive_df.rename(columns={0: "words", 1: "Coefficients"}, inplace=True)

    negative_df.set_index("words", inplace=True)
    positive_df.set_index("words", inplace=True)

    np.random.seed(19680801)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharex=False, sharey=False, figsize=(12, 6), frameon=True
    )
    ax1 = negative_df.plot.barh(color="#007acc", ax=ax1)
    ax2 = positive_df.plot.barh(color="orange", ax=ax2)

    ax1.set_title("Most negative words", size=10)
    ax2.yaxis.tick_right()
    ax2.set_title("Most positive words", size=10)

    # Place legends in the bottom corner
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")

    fig.suptitle("Word Feature Analysis")


# Example usage
word_feature_plot(w2v_model_2, rfc)

# ---------------------------------------------------------------------------------------------------------
# Explore model probability distributions
# ---------------------------------------------------------------------------------------------------------


def plot_distribution(model):
    """_summary_

    Args:
        model (_type_): _description_
    """
    if model == lsvc:
        # Use decision function and sigmoid to obtain probabilities
        model_scores = model.decision_function(X_test)
        model_probs = 1 / (1 + np.exp(-model_scores))
    else:
        # Predict probabilities
        model_probs = model.predict_proba(X_test)[:, 1]

    # Calculate histogram data
    _, bins = np.histogram(model_probs, bins=10)

    # Plot histogram for the current model with a different color
    plt.hist(model_probs, bins=bins, alpha=0.8)
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Samples")
    plt.title(f"{model}")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()


# Example usage
plot_distribution(lgcv)
