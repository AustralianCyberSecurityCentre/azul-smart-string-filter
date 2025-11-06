# Smart String filter

Uses an AI model to find interesting strings in a PE file.

## Installation

```bash
pip install azul-smart-string-filter
```

## Usage

A default entrypoint has been defined in `setup.py`, which will run `main()` in `azul_smart_string_filter/train/main.py`  
There are two functions that can be called: train and tune.
Example: `azul-smart-string-filter trainmodel RF f1 RS`, `azul-smart-string-filter tune RF f1 RS`
The above example uses the parameters model = RF = RandomForest, score=f1, Search type = RS  
More details on these below.

```bash
$ azul-smart-string-filter
Pavlov probably thought about feeding his dogs every time someone rang a bell.
```

This is a string filter using an AI model to filter strings. There is no complexity around the classification of good strings (those that will be shown), and bad strings (those that will be filtered). It was decided to classify human-readable strings as good (strings containing English, and in some cases, patterns that stood out), and the rest as bad.
This process could classify passwords and other valuable strings as bad as they are generally random looking.
More work could be done on fine-tuning the classification process. e.g. bring entropy in as a factor when tuning and training the ai model.

Data was taken from the first 1000 (if there were over 1000 strings in the file) strings from Windows PE binary files. Using this method, 25,000 good strings were collected along with 25,000 bad strings from roughly 70 different files.

After extensive testing, the Random Forest model was found to be the best performing model.  
To determine this a number of random files were taken from Azul, the strings were run through each model, and the True positive and True negatives were counted and divided by the total number of strings in the file.  
The performance on some non-windows files was questionable.  
In future, multiple models might be needed to tackle different file types, or the training data needs to be extended to not just Windows PE files.

Other models that were considered were Gradient Boosting (GB), K-Nearest Neighbours (KNN), Logistic Regression (LR), Naive Bayes Multinomial (NB) and Support Vector Machine (SVM).

## Training

Use the Scikit library for the AI model. This was chosen due to it being popular in python development.
The training process first starts with tuning. The tuning process consists of defining the hyperparameters you want to optimise for. These were the hyperparameters chosen for RF:

```python
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
```

Increasing the number of hyperparameters and values increases the tuning time. The above hyperparameters were chosen to balance accuracy and tuning time.
You can use the command line interface to tune like this:
`azul-smart-string-filter tune RF accuracy GS`. Example of tuning the Random Forest model, with the score type accuracy, and search type GridSearch.
The hyperparameter report will be saved as:
./models/RF/parameters/GS/RF_accuracy_best_parameters_report_GS.txt for the above example. Different models etc. will follow a similar pattern.

To train the model you can use the command line interface like this:
`azul-smart-string-filter trainmodel` RF accuracy GS. Example of training the Random Forest model, with the score type accuracy, and search type GridSearch.
The model (along with the vectorizer) will be saved as:
./models/RF/models/GS/RF_accuracy_classifier_model_GS.onnx
./models/RF/models/GS/RF_accuracy_tfidf_vectorizer_GS.json

The ability to train different models has been preserved (even underperforming models) in case there is a
change to training data / implementation that could necessitate testing of different models again.

## Score types

- Accuracy: Proportion of correctly classified instances out of the total instances.
- Precision: Proportion of true positive predictions out of all positive predictions. It measures the model's ability to correctly identify positive instances.
- Recall (Sensitivity): Proportion of true positive predictions out of all actual positive instances. It measures the model's ability to find all positive instances.
- F1 Score: Harmonic mean of precision and recall. It provides a balance between precision and recall.

The score types that can be input are "f1", "precision", "recall" and "accuracy"

Then you must decide whether to tune with random search, or grid search.

## Random search

RandomizedSearch (or RandomizedSearchCV in scikit-learn) is an alternative to GridSearchCV for hyperparameter tuning. Instead of searching exhaustively through all possible combinations of hyperparameters, RandomizedSearchCV samples a fixed number of hyperparameter combinations from a specified distribution. This can be more efficient and often leads to finding good hyperparameters in less time compared to an exhaustive grid search.
Random search performs better and you can therefore use a larger data set to tune it with.

## Grid search

GridSearch is a method used in machine learning to systematically search for the best hyperparameters for a given
model. 
It performs an exhaustive search over a specified parameter grid to find the combination of parameters that results
in the highest model performance. GridSearch typically uses cross-validation to evaluate the performance
of each combination of hyperparameters. This involves splitting the training data into multiple folds
and using some folds for training and others for validation.

Tuning uses CPU and supports multiprocessing, so the more CPU's you have the faster it will be.
Warning: Some of these models cannot be tuned in a reasonable timeframe with a standard VM. A larger VM with 64 CPUS was used for a number of days to get all the optimal hyperparameter combinations for each model.

## How the filter works

The filter takes a list of strings, runs each string through the model, which returns 1 for good, or 0 for bad. It populates a list of bools and returns it to the calling function.
For example: this list is fed into the filter ['asdfasdf', 'Not random', 'ad#F$', 'filter']
And the filter returns a list like this: [False, True, False, True]

To use the function to filter a list of strings you follow these steps:
GSF = SmartStringFilter()
list_of_bools = GSF.find_legible_strings(string_list)

All of the models and best hyperparameters are located in ./models/ with the format ./model_name/models/search_type/model_name_scoretype_classifier_model_search_type.joblib
For every model there is an associated vectorizer also located in ./models/ with the format ./model_name/models/search_type/model_name_scoretype_tfidf_vectorizer_searchtype.joblib
So the SVM model, tuned and trained with score type f1, and tuned with random search would be stored at:
model: ./models/SMV/models/RS/SVM_f1_classifier_model_RS.joblib
vectorizer: ./models/SMV/models/RS/SVM_f1_tfidf_vectorizer_RS.joblib

The hyperparameters and performance reports are kept in ./models in the format ./model_name/parameters/search_type/model_name_scoretype_best_parameters_report_search_type.txt and ./model_name/parameters/search_type/model_name_scoretype_classification_report_search_type.txt.
So the SVM model, tuned with score type f1 and using randomsearch would be stored at:
./models/SVM/parameters/RS/SVM_f1_best_parameters_report_RS.txt
./models/SVM/parameters/RS/SVM_f1_classification_report_RS.txt
