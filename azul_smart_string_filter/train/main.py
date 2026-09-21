"""This module is used to tune and train AI models."""

import ast
import json
import os
import time

import click
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

RANDOM_SEARCH_ITERATIONS = 50


@click.group()
def cli():
    """Cli method for main."""
    pass


@cli.command()
@click.argument("model", type=click.Choice(["RF", "GB", "SVM", "KNN", "LR", "NB"]))
@click.argument("score", type=click.Choice(["f1", "recall", "precision", "accuracy"]))
@click.argument("search", type=click.Choice(["RS", "GS"]))
@click.argument("model_type")
def tune(model, score, search, model_type):
    """Cli method for tuning models."""
    best_parameter_estimator(model, score, search, model_type)


@cli.command()
@click.argument("model", type=click.Choice(["RF", "GB", "SVM", "KNN", "LR", "NB"]))
@click.argument("score", type=click.Choice(["f1", "recall", "precision", "accuracy"]))
@click.argument("search", type=click.Choice(["RS", "GS"]))
@click.argument("model_type")
def trainmodel(model, score, search, model_type):
    """Cli method for training models."""
    train(model, score, search, model_type)


def normalise_model_type(model_type: str) -> str:
    """Normalise and validate the dataset/model type used in filenames."""
    model_type = model_type.strip().lower()
    if not model_type or not model_type.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid model type: {model_type!r}")
    return model_type


def load_training_strings(model_type: str):
    """Load, clean, and deduplicate the good and bad training strings."""
    data_dir = "azul_smart_string_filter"

    def load_file(filename):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8", errors="ignore") as file:
            return list(dict.fromkeys(line.strip() for line in file if line.strip()))

    good_strings = load_file(f"good_{model_type}.txt")
    bad_strings = load_file(f"bad_{model_type}.txt")

    overlap = set(good_strings).intersection(bad_strings)
    if overlap:
        examples = ", ".join(repr(value) for value in sorted(overlap)[:5])
        raise ValueError(f"Found {len(overlap)} strings labelled as both good and bad. Examples: {examples}")

    if not good_strings or not bad_strings:
        raise ValueError("Both the good and bad training files must contain at least one string")

    return good_strings, bad_strings


def classifier_parameter_grid(param_grid):
    """Prefix classifier parameters for use in a scikit-learn Pipeline."""
    if isinstance(param_grid, list):
        return [{f"classifier__{name}": values for name, values in grid.items()} for grid in param_grid]
    return {f"classifier__{name}": values for name, values in param_grid.items()}


def save_best_parameters(filename: str, parameters: dict):
    """Save classifier parameters in a safely reloadable format."""
    classifier_parameters = {name.removeprefix("classifier__"): value for name, value in parameters.items()}
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(classifier_parameters, file, indent=2, sort_keys=True)


def load_best_parameters(filename: str):
    """Load JSON parameters, with support for reports written by older versions."""
    with open(filename, "r", encoding="utf-8") as file:
        parameter_string = file.read()

    try:
        parameters = json.loads(parameter_string)
    except json.JSONDecodeError:
        parameters = ast.literal_eval(parameter_string)

    if not isinstance(parameters, dict):
        raise ValueError(f"Invalid parameter report: {filename}")

    return {name.removeprefix("classifier__").removeprefix("model__"): value for name, value in parameters.items()}


def best_parameter_estimator(model: str, score_type: str, search_type: str, model_type: str):
    """Find the best hyperparameters for your model."""
    model_type = normalise_model_type(model_type)
    good_strings, bad_strings = load_training_strings(model_type)

    X = good_strings + bad_strings
    y = [1] * len(good_strings) + [0] * len(bad_strings)  # 0 for bad, 1 for good.

    # Split raw strings before fitting TF-IDF so the held-out test data does not
    # influence the vocabulary or IDF values.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # model_parameter dict with model as key and value is tuple.
    # containing hyperparamater ranges and classifier object.
    model_parameter = {
        "SVM": (
            [
                {
                    "kernel": ["linear"],
                    "C": [0.1, 1, 10, 100],
                    "class_weight": [None, "balanced"],
                    "shrinking": [True, False],
                    "probability": [False],
                    "tol": [1e-3],
                    "max_iter": [1000, 5000],
                },
                {
                    "kernel": ["rbf"],
                    "C": [0.1, 1, 10, 100],
                    "gamma": [0.001, 0.01, 0.1, 1],
                    "class_weight": [None, "balanced"],
                    "shrinking": [True, False],
                    "probability": [False],
                    "tol": [1e-3],
                    "max_iter": [1000, 5000],
                },
                {
                    "kernel": ["poly"],
                    "C": [0.1, 1, 10, 100],
                    "gamma": [0.001, 0.01, 0.1, 1],
                    "degree": [2, 3, 4],
                    "coef0": [0, 0.1, 0.5, 1],
                    "class_weight": [None, "balanced"],
                    "shrinking": [False],
                    "probability": [False],
                    "tol": [1e-3],
                    "max_iter": [1000, 5000],
                },
                {
                    "kernel": ["sigmoid"],
                    "C": [0.1, 1, 10, 100],
                    "gamma": [0.001, 0.01, 0.1, 1],
                    "coef0": [0, 0.1, 0.5, 1],
                    "class_weight": [None, "balanced"],
                    "shrinking": [True, False],
                    "probability": [False],
                    "tol": [1e-3],
                    "max_iter": [1000, 5000],
                },
            ],
            SVC(),
        ),
        "NB": (
            {
                "alpha": [
                    0.1,
                    1.0,
                    10.0,
                ],  # Specify the alpha parameter for MultinomialNB.
                "fit_prior": [
                    True,
                    False,
                ],  # Specify the fit_prior parameter for MultinomialNB.
            },
            MultinomialNB(),
        ),
        "LR": (
            {
                "max_iter": [1000, 5000],  # Increase the number of iterations.
                "penalty": ["l1", "l2"],  # Regularization type: L1 or L2.
                "C": [
                    0.001,
                    0.01,
                    0.1,
                    1,
                    10,
                    100,
                ],  # Inverse of regularization strength.
                "solver": [
                    "liblinear",
                    "saga",
                ],  # Algorithm to use in the optimization problem.
                "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria.
                "fit_intercept": [True, False],  # Whether to add a bias term.
                "class_weight": [None, "balanced"],  # Balances class weights.
            },
            LogisticRegression(),
        ),
        "KNN": (
            {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "algorithm": ["brute"],
                "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance.
                "leaf_size": [20, 30, 40],  # Leaf size passed to BallTree or KDTree.
                "metric": [
                    "minkowski",
                    "manhattan",
                    "euclidean",
                ],  # Distance metric to use.
                "n_jobs": [-1],  # Use all available CPUs.
            },
            KNeighborsClassifier(),
        ),
        "RF": (
            {
                "n_estimators": [100, 200, 300],  # Number of trees in the forest.
                "max_depth": [None, 10, 20],  # Maximum depth of the tree.
                "min_samples_split": [
                    2,
                    5,
                    10,
                ],  # Minimum number of samples required to split an internal node.
                "min_samples_leaf": [
                    1,
                    2,
                    4,
                ],  # Minimum number of samples required to be at a leaf node.
                "bootstrap": [
                    True,
                    False,
                ],  # Whether bootstrap samples are used when building trees.
                "max_features": [
                    "sqrt",
                    "log2",
                    None,
                ],  # Number of features to consider when looking for the best split.
                "max_leaf_nodes": [None, 10, 20, 30],  # Maximum number of leaf nodes.
            },
            RandomForestClassifier(),
        ),
        "MLP": (
            {
                "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 100)],
                "activation": ["relu", "tanh"],
                "solver": ["adam", "sgd"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate": ["constant", "adaptive"],
            },
            MLPClassifier(),
        ),
        "GB": (
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.1, 0.05, 0.01],
                "max_depth": [3, 4, 5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "subsample": [
                    0.8,
                    1.0,
                ],  # Fraction of samples used for fitting the individual learners.
            },
            GradientBoostingClassifier(),
        ),
    }

    parameters = model_parameter[model]
    if not parameters:
        print("invalid model")
        return
    param_grid = classifier_parameter_grid(parameters[0])
    clf = parameters[1]
    pipeline = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 6),
                    lowercase=False,
                    min_df=2,
                    dtype=np.float32,
                ),
            ),
            ("classifier", clf),
        ]
    )

    base_dir = os.path.join("models", model, "parameters")
    os.makedirs(os.path.join(base_dir, "RS"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "GS"), exist_ok=True)
    if search_type == "RS":
        # Perform random search.
        # RandomizedSearch (or RandomizedSearchCV in scikit-learn) is
        # an alternative to GridSearchCV for hyperparameter tuning.
        # Instead of searching exhaustively through all possible combinations
        # of hyperparameters, RandomizedSearchCV samples a fixed number of
        # hyperparameter combinations from a specified distribution.
        # This can be more efficient and often leads to finding good hyperparameters
        # in less time compared to an exhaustive grid search.
        random_search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_grid,
            n_iter=RANDOM_SEARCH_ITERATIONS,
            cv=5,
            scoring=score_type,
            random_state=42,
            n_jobs=-1,
            verbose=3,
        )
        print("Tuning with random search.")
        random_search.fit(X_train, y_train)
        # Print the best parameters found.
        print("Best Parameters:", random_search.best_params_)

        save_best_parameters(
            os.path.join(base_dir, "RS", f"{model}_{score_type}_best_parameters_report_RS.txt"),
            random_search.best_params_,
        )
        # Evaluate the best model on the test set.
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(classification_report(y_test, y_pred))
        # Save the best parameters.
        with open(
            os.path.join(base_dir, "RS", f"{model}_{score_type}_classification_report_RS.txt"),
            "w",
        ) as f:
            f.write(classification_report(y_test, y_pred))
    elif search_type == "GS":
        # Perform grid search with cross-validation
        # GridSearch is a method used in machine learning to
        # systematically search for the best hyperparameters
        # for a given model. It performs an exhaustive search
        # over a specified parameter grid to find the combination
        # of parameters that results in the highest model performance.
        # GridSearch typically uses cross-validation to evaluate the performance
        # of each combination of hyperparameters. This involves splitting the training
        # data into multiple folds and using some folds for training and others for validation.
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=5,
            scoring=score_type,
            n_jobs=-1,
            verbose=3,
        )
        print("Tuning with grid search.")
        grid_search.fit(X_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        # Save the best parameters.
        save_best_parameters(
            os.path.join(base_dir, "GS", f"{model}_{score_type}_best_parameters_report_GS.txt"),
            grid_search.best_params_,
        )
        # Evaluate the best model on the test set.
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(classification_report(y_test, y_pred))
        with open(
            os.path.join(base_dir, "GS", f"{model}_{score_type}_classification_report_GS.txt"),
            "w",
        ) as f:
            f.write(classification_report(y_test, y_pred))
    else:
        print("Invalid search type. Search types are GS or RS")


def train(model_name: str, score_type: str, search: str, model_type: str):
    """Train your desired model with training data."""
    model_type = normalise_model_type(model_type)

    if search == "RS":
        print(f"Training: {model_name} with {score_type} random search")
    elif search == "GS":
        print(f"Training: {model_name} with {score_type} grid search")
    else:
        print("invalid search parameter. User GS or RS")
        return

    good_strings, bad_strings = load_training_strings(model_type)

    # Use Term Frequency-Inverse Document Frequency vectorizer to transform the raw text into
    # a numerical representation that can be used for training by machine learning algorithms.
    # Analyzer=char: vectorizer considers character sequences across the full string.
    # ngram_range: vectorizer considers all sequences of 2 to 6 consecutive items.
    # lowercase=False preserves casing such as CamelCase and ALL_CAPS identifiers.
    # min_df=2 ignores n-grams that occur in only one training string.
    # in the text. These items will be characters based on the analyzer
    # for the word example:
    # 2-grams: "ex", "xa", "am", "mp", "pl", "le"
    # 3-grams: "exa", "xam", "amp", "mpl", "ple"
    # 4-grams: "exam", "xamp", "ampl", "mple"
    # 5-grams: "examp", "xampl", "ample"
    # 6-grams: "exampl", "xample".
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 6),
        lowercase=False,
        min_df=2,
        dtype=np.float32,
    )
    training_strings = good_strings + bad_strings
    X = vectorizer.fit_transform(training_strings)
    y = [1] * len(good_strings) + [0] * len(bad_strings)  # 0 for bad, 1 for good.

    # Get best parameters for model
    # The best parameters for each model were determined by gridsearch and randomsearch
    # This can take a LONG time. 64 cpu VM was used to find the best parameters for each model.
    base_dir = os.path.join("models", model_name)
    if search == "RS":
        parameter_filename = os.path.join(
            base_dir,
            "parameters",
            "RS",
            f"{model_name}_{score_type}_best_parameters_report_RS.txt",
        )
    elif search == "GS":
        parameter_filename = os.path.join(
            base_dir,
            "parameters",
            "GS",
            f"{model_name}_{score_type}_best_parameters_report_GS.txt",
        )
    else:
        raise ValueError("Invalid search parameter. Use GS or RS")

    parameters = load_best_parameters(parameter_filename)
    print("Using best parameters: ", parameters)
    best_parameters = parameters

    # Create the model using the best parameters in the constructor.
    if model_name == "SVM":
        model = SVC(**best_parameters)
    elif model_name == "NB":
        model = MultinomialNB(**best_parameters)
    elif model_name == "LR":
        model = LogisticRegression(**best_parameters)
    elif model_name == "KNN":
        model = KNeighborsClassifier(**best_parameters)
    elif model_name == "RF":
        model = RandomForestClassifier(**best_parameters)
    elif model_name == "MLP":
        model = MLPClassifier(**best_parameters)
    elif model_name == "GB":
        model = GradientBoostingClassifier(**best_parameters)
    else:
        print("invalid model")
        return

    # train the model with the data.
    start_time = time.time()
    model.fit(X, y)
    end_time = time.time()

    # Output the training time.
    print(f"Training time: {end_time - start_time:.2f} seconds")
    # Save the model and the vectorizer to files.
    if search == "RS":
        model_filename = os.path.join(
            base_dir,
            "model",
            "RS",
            f"{model_name}_{model_type}_{score_type}_classifier_model_RS.onnx",
        )
        vectorizer_filename = os.path.join(
            base_dir,
            "model",
            "RS",
            f"{model_name}_{model_type}_{score_type}_tfidf_vectorizer_RS.json",
        )
    elif search == "GS":
        model_filename = os.path.join(
            base_dir,
            "model",
            "GS",
            f"{model_name}_{model_type}_{score_type}_classifier_model_GS.onnx",
        )
        vectorizer_filename = os.path.join(
            base_dir,
            "model",
            "GS",
            f"{model_name}_{model_type}_{score_type}_tfidf_vectorizer_GS.json",
        )

    # Define the intial type for input.
    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]

    # Convert the pipeline to ONNX format.
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)

    # Save the ONNX model to a file.
    with open(model_filename, "wb") as f:
        f.write(onnx_model.SerializePartialToString())

    vectorizer_json = {
        "vocabulary_": vectorizer.vocabulary_,
        "idf_": vectorizer.idf_.tolist(),
        "ngram_range": vectorizer.ngram_range,
        "analyzer": vectorizer.analyzer,
        "lowercase": vectorizer.lowercase,
        "min_df": vectorizer.min_df,
    }

    # Save the vectorizer as JSON
    with open(vectorizer_filename, "w") as f:
        json.dump(vectorizer_json, f)


if __name__ == "__main__":
    """For command line arguments."""
    cli()
