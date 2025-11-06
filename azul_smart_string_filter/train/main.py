"""This module is used to tune and train AI models."""

import json
import os
import time

import click
from scipy import sparse
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
from sklearn.svm import SVC


@click.group()
def cli():
    """Cli method for main."""
    pass


@cli.command()
@click.argument("model", type=click.Choice(["RF", "GB", "SVM", "KNN", "LR", "NB"]))
@click.argument("score", type=click.Choice(["f1", "recall", "precision", "accuracy"]))
@click.argument("search", type=click.Choice(["RS", "GS"]))
def tune(model, score, search):
    """Cli method for tuning models."""
    best_parameter_estimator(model, score, search)


@cli.command()
@click.argument("model", type=click.Choice(["RF", "GB", "SVM", "KNN", "LR", "NB"]))
@click.argument("score", type=click.Choice(["f1", "recall", "precision", "accuracy"]))
@click.argument("search", type=click.Choice(["RS", "GS"]))
def trainmodel(model, score, search):
    """Cli method for training models."""
    train(model, score, search)


def best_parameter_estimator(model: str, score_type: str, search_type: str):
    """Find the best hyperparameters for your model."""
    with open(os.path.join("azul_smart_string_filter", "good.txt"), "r") as f:
        good_strings = [line.strip() for line in f]
    with open(os.path.join("azul_smart_string_filter", "bad.txt"), "r") as f:
        bad_strings = [line.strip() for line in f]

    # Create a vectorizer and vectorize the data.
    # A vectorizer, in the context of natural language processing (NLP),
    # refers to a tool or technique that converts textual data into numerical vectors.
    # These vectors can then be used as input to machine learning algorithms for
    # tasks such as classification, regression, clustering, or any other type
    # of analysis that requires numerical input.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6))
    X_good = vectorizer.fit_transform(good_strings)
    X_bad = vectorizer.transform(bad_strings)

    # Use a sparse matrix for X.
    # When you vectorize text data using methods like TfidfVectorizer,
    # the resulting matrix typically has many zero values because each
    # document will only contain a subset of the total vocabulary.
    # Sparse Representation:
    # Instead of storing all elements (including zeros) in a dense format,
    # sparse matrices store only the non-zero elements along with their indices.
    # This significantly reduces memory usage and speeds up operations for matrices with many zero values.
    X = sparse.vstack([X_good, X_bad])
    y = [1] * len(good_strings) + [0] * len(bad_strings)  # 0 for bad, 1 for good.

    # Split the data into training and testing sets (don't test with data you used to train with).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
                "alpha": [0.1, 1.0, 10.0],  # Specify the alpha parameter for MultinomialNB.
                "fit_prior": [True, False],  # Specify the fit_prior parameter for MultinomialNB.
            },
            MultinomialNB(),
        ),
        "LR": (
            {
                "max_iter": [1000, 5000],  # Increase the number of iterations.
                "penalty": ["l1", "l2"],  # Regularization type: L1 or L2.
                "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength.
                "solver": ["liblinear", "saga"],  # Algorithm to use in the optimization problem.
                "tol": [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criteria.
                "fit_intercept": [True, False],  # Whether to add a bias term.
                "class_weight": [None, "balanced"],  # Balances class weights.
                "multi_class": ["auto", "ovr", "multinomial"],  # Multi-class handling strategy.
            },
            LogisticRegression(),
        ),
        "KNN": (
            {
                "n_neighbors": [3, 5, 7, 10],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "p": [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance.
                "leaf_size": [20, 30, 40],  # Leaf size passed to BallTree or KDTree.
                "metric": ["minkowski", "chebyshev", "mahalanobis"],  # Distance metric to use.
                "n_jobs": [-1],  # Use all available CPUs.
            },
            KNeighborsClassifier(),
        ),
        "RF": (
            {
                "n_estimators": [100, 200, 300],  # Number of trees in the forest.
                "max_depth": [None, 10, 20],  # Maximum depth of the tree.
                "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node.
                "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node.
                "bootstrap": [True, False],  # Whether bootstrap samples are used when building trees.
                "max_features": [
                    "auto",
                    "sqrt",
                    "log2",
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
                "subsample": [0.8, 1.0],  # Fraction of samples used for fitting the individual learners.
            },
            GradientBoostingClassifier(),
        ),
    }

    # Pipeline for text vectorization.
    parameters = model_parameter[model]
    if not parameters:
        print("invalid model")
        return
    param_grid = parameters[0]
    clf = parameters[1]

    base_dir = os.path.join("models", model, "parameters")
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
            clf, param_distributions=param_grid, n_iter=1, cv=5, random_state=42, n_jobs=-1, verbose=3
        )
        print("Tuning with random search.")
        random_search.fit(X_train, y_train)
        # Print the best parameters found.
        print("Best Parameters:", random_search.best_params_)

        with open(os.path.join(base_dir, "RS", f"{model}_{score_type}_best_parameters_report_RS.txt"), "w") as f:
            f.write(str(random_search.best_params_))
        # Evaluate the best model on the test set.
        best_model = random_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(classification_report(y_test, y_pred))
        # Save the best parameters.
        with open(os.path.join(base_dir, "RS", f"{model}_{score_type}_classification_report_RS.txt"), "w") as f:
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
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, scoring=score_type, n_jobs=-1, verbose=3)
        print("Tuning with grid search.")
        grid_search.fit(X_train, y_train)
        print("Best Parameters:", grid_search.best_params_)
        # Save the best parameters.
        with open(os.path.join(base_dir, "GS", f"{model}_{score_type}_best_parameters_report_GS.txt"), "w") as f:
            f.write(str(grid_search.best_params_))
        # Evaluate the best model on the test set.
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(classification_report(y_test, y_pred))
        with open(os.path.join(base_dir, "GS", f"{model} _{score_type}_classification_report_GS.txt"), "w") as f:
            f.write(classification_report(y_test, y_pred))
    else:
        print("Invalid search type. Search types are GS or RS")


def train(model_name: str, score_type: str, search: str):
    """Train your desired model with training data."""
    if search == "RS":
        print(f"Training: {model_name} with {score_type} random search")
    elif search == "GS":
        print(f"Training: {model_name} with {score_type} grid search")
    else:
        print("invalid search parameter. User GS or RS")
        return

    # Load the sample of "good" strings.
    with open(os.path.join("azul_smart_string_filter", "good.txt"), "r") as f:
        good_strings = [line.strip() for line in f]
    # Load the sample of "bad" strings
    with open(os.path.join("azul_smart_string_filter", "bad.txt"), "r") as f:
        bad_strings = [line.strip() for line in f]

    # Use Term Frequency-Inverse Document Frequency vectorizer to transform the raw text into
    # a numerical representation that can be used for training by machine learning algorithms.
    # Analyser=char_wb: vectorizer will consider character sequences
    # ngram_range: vectorizer will consider all possible sequences of 4 to 6 consecutive items
    # in the text. These items will be characters based on the analyzer
    # for the word example:
    # 4-grams: "exam", "xamp", "ampl", "mple"
    # 5-grams: "examp", "xampl", "ample"
    # 6-grams: "exampl", "xample".
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 6))
    X_good = vectorizer.fit_transform(good_strings)
    X_bad = vectorizer.transform(bad_strings)

    # combine the good and bad data into a sparse matrix for training.
    X = sparse.vstack([X_good, X_bad])
    # add labels for good (0) and bad(1) data.
    y = [1] * len(good_strings) + [0] * len(bad_strings)  # 0 for bad, 1 for good.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Get best parameters for model
    # The best parameters for each model were determined by gridsearch and randomsearch
    # This can take a LONG time. 64 cpu VM was used to find the best parameters for each model.
    base_dir = os.path.join("models", model_name)
    if search == "RS":
        with open(
            os.path.join(base_dir, "parameters", "RS", f"{model_name}_{score_type}_best_parameters_report_RS.txt"), "r"
        ) as file:
            parameter_string = file.readline()
    elif search == "GS":
        with open(
            os.path.join(base_dir, "parameters", "GS", f"{model_name}_{score_type}_best_parameters_report_GS.txt"), "r"
        ) as file:
            parameter_string = file.readline()
    else:
        print("Invalid search parameter")

    # Additional formatting for best parameter string so that scikit can
    # recognise it:
    # Remove the leading and trailing characters (e.g., '{', '}', '\n').
    parameter_string = parameter_string.strip("{}\n")
    # Split the line into parameter-value pairs.
    param_value_pairs = parameter_string.split(", ")
    # Load the parameters from the parameter string.
    parameters = {}
    for pair in param_value_pairs:
        param, value = pair.split(": ")
        param_name = param.replace(model_name + "__", "")
        # Convert the value to the appropriate type (e.g., int, bool).
        if value.isdigit():
            value = int(value)
            parameters[param_name] = value
            continue
        try:
            value = float(value)
            parameters[param_name] = value
            continue
        except ValueError:
            pass

        if value == "None":
            parameters[param_name] = None
            continue

        if value.lower() == "true" or value.lower() == "false":
            value = bool(value)

        parameters[param_name] = value

    # Remove extra quotes from keys.
    parameters = {
        key.strip("'\""): value.strip("'\"") if isinstance(value, str) else value for key, value in parameters.items()
    }
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
    model.fit(X_train, y_train)
    end_time = time.time()

    # Output the training time.
    print(f"Training time: {end_time - start_time:.2f} seconds")
    # Save the model and the vectorizer to files.
    if search == "RS":
        model_filename = os.path.join(base_dir, "model", "RS", f"{model_name}_{score_type}_classifier_model_RS.onnx")
        vectorizer_filename = os.path.join(
            base_dir, "model", "RS", f"{model_name}_{score_type}_tfidf_vectorizer_RS.json"
        )
    elif search == "GS":
        model_filename = os.path.join(base_dir, "model", "GS", f"{model_name}_{score_type}_classifier_model_GS.onnx")
        vectorizer_filename = os.path.join(
            base_dir, "model", "GS", f"{model_name}_{score_type}_tfidf_vectorizer_GS.json"
        )

    # Define the intial type for input.
    initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]

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
    }

    # Save the vectorizer as JSON
    with open(vectorizer_filename, "w") as f:
        json.dump(vectorizer_json, f)


if __name__ == "__main__":
    """For command line arguments."""
    cli()
