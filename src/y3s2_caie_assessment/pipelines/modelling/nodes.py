import pandas as pd
import datetime as datetime
from typing import List, Dict
import numpy as np
import lightgbm as lgb

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def date_difference(row: pd.Series, params: dict) -> int:
    """
    Calculate the difference in days between two dates.

    Parameters:
        row (pd.Series): The row from the dataset.
        mapping (dict): A dictionary containing:
            - 'actual_date': Column name for actual delivery first date.
            - 'estimated_date': Column name for the estimated delivery date.
            - 'actual_date_format': Format of the actual delivery date.
            - 'estimated_date_format': Format of the estimated delivery date.

    Returns:
        int: The difference in days between the two dates.
    """
    try:
        # Identify the 2 dates
        date1 = row[params['actual_date']]
        date2 = row[params['estimated_date']]
        # Check if there are any missing dates
        if pd.isna(date1) or pd.isna(date2):
            return None # Returns None when its not delivered yet
        
        # Parse the dates
        d1 = datetime.strptime(date1, params['actual_date_format'])
        d2 = datetime.strptime(date2, params['estimated_date_format'])

        # Calculate the difference
        day_difference = (d2 - d1).days
        return day_difference
    
    except ValueError as ve:
        raise ValueError(f"Invalid date or format: {ve}")
    
    
def feature_engineering(df: pd.DataFrame, new_feature: str, function_name: str, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create a new feature based on the features in the row.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        new_feature (str): The column name of the new feature to be created.
        function_name (str): The name of the function to be applied in order to create the feature.
        mapping (List[str]): Mapping column names to the parameter names used by the function. Provides other arguments for the function.

    Returns:
        pd.DataFrame: The DataFrame with new features.
    """
    try:
        # Obtain the function based on name
        function = globals()[function_name]

        # Check if the function is valid
        if not callable(function):
            raise ValueError(f"Function is invalid or not callable")
        
        # Initialise the feature engineered DataFrame
        feature_engineered_df = df.copy()
        
        # Apply the function to each row and create the new feature
        feature_engineered_df[new_feature] = df.apply(lambda row: function(row, mapping), axis=1)
        return feature_engineered_df

    except ValueError as ve:
        # Show error
        print(f"ValueError: {ve}")
        return df

    except Exception as e:
        # Show the error
        print(f"An error occurred during feature engineering: {e}")
        return df
    
    
def standardisation_and_encoding(df: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Encodes the categorical data into One Hot Encoded vectors and standardises the scaling of numerical data
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (List[str]): The list of numerical features in the DataFrame.
        categorical_cols (List[str]): The list of categorical features in the DataFrame.

    Returns:
        feature_encoded_df (pd.DataFrame): The DataFrame after feature encoding and standardisation.
    """
    try:
        # Combine the numerical and categorical column lists
        columns = numerical_cols + categorical_cols

        # Check if all columns are included in the standardisation and encoding
        if set(columns) != set(df.columns):
            raise ValueError(f"Not all columns were included: {ve}")

        # Initialize encoders and scalers
        one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
        scaler = StandardScaler()

        # Process categorical columns
        encoded_categorical_data = one_hot_encoder.fit_transform(df[categorical_cols])
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical_data, 
            columns=one_hot_encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )

        # Process numerical columns
        scaled_numerical_data = scaler.fit_transform(df[numerical_cols])
        scaled_numerical_df = pd.DataFrame(
            scaled_numerical_data,
            columns=numerical_cols,
            index=df.index
        )

        # Combine the processed data 
        feature_encoded_df = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)
        return feature_encoded_df
    
    except ValueError as ve:
        # Show error
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        # Show the error
        print(f"An unexpected error occurred: {e}")
        return df



def train_test_split(df: pd.DataFrame, split_parameters: dict, random_state: int):
    """
    Uses stratified shuffle split to split the dataframe into evenly distributed train and test datasets for modelling

    Args:
        df (pd.DataFrame): The input feature encoded dataframe to be split for modelling

    Returns:
        X_train: The features used for training
        y_train: The labels of the dataset used for training
        X_test: The features used for testing
        y_test: The labels of the dataset used for testing
    """
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    sss = StratifiedShuffleSplit(n_splits=split_parameters['n_splits'], 
                                 test_size=split_parameters['test_size'], 
                                 random_state=random_state)
    for train_index, test_index in sss.split(X, y): 
        X_train, X_test = X.iloc[train_index], X.iloc[test_index] 
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, y_train, X_test, y_test


def train_random_forest(X_train, y_train, model_params, kfold_params):
    """
    Train a Random Forest classifier using K-Fold cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for RandomForestClassifier.
        kfold_params (int): Parameters for StratifiedKFold.
    """
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_fold_train, y_fold_train)
        
        val_pred = model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

        # Store the model along with the associated indicies for train val split
        models.append(model)

        # Display fold metrics
        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Select the best model with highest validation accuracy to represent the unoptimized model
    best_unoptimized_model_idx = np.argmax(val_metrics["accuracy"])
    best_unoptimized_model = models[best_unoptimized_model_idx]
    print(f"\nBest Unoptimized Model Fold: {best_unoptimized_model_idx + 1}")

    # Display the aggregated metrics of the unoptimized models
    avg_val_metrics = {
        "accuracy": np.mean(val_metrics["accuracy"]),
        "f1": np.mean(val_metrics["f1"]),
        "precision": np.mean(val_metrics["precision"]),
        "recall": np.mean(val_metrics["recall"])
    }
    std_val_metrics = {
        "accuracy": np.std(val_metrics["accuracy"]),
        "f1": np.std(val_metrics["f1"]),
        "precision": np.std(val_metrics["precision"]),
        "recall": np.std(val_metrics["recall"])
    }
    print("\nAggregated Metrics of Unoptimized RandomForest Models:")
    print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
    print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
    print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
    print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

    # Save aggregated results into a dataframe for passing downstream to accumulate the other val metrics of other models
    results_df = pd.DataFrame({
        "model_type": ["Unoptimized RF"],
        "avg_accuracy": [avg_val_metrics["accuracy"]],
        "avg_f1": [avg_val_metrics["f1"]],
        "avg_precision": [avg_val_metrics["precision"]],
        "avg_recall": [avg_val_metrics["recall"]],
        "std_accuracy": [std_val_metrics["accuracy"]],
        "std_f1": [std_val_metrics["f1"]],
        "std_precision": [std_val_metrics["precision"]],
        "std_recall": [std_val_metrics["recall"]]
    })

    # Return the metrics and best unoptimized model
    return results_df, best_unoptimized_model


def optimize_random_forest(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize a Random Forest classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for RandomForestClassifier.
        kfold_params (dict): Parameters for StratifiedKFold.
    """
    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    # Parameters to perform the grid search on
    grid_params = {
        "max_depth": [1, 2, 4, 8], # Max depth of trees to control overfitting
        "n_estimators": [100, 200, 300], # Number of trees in the forest, higher can improve accuracy
        "min_samples_split": [5, 10, 15], # Minimum samples to split an internal node, higher value reduces overfitting
        "min_samples_leaf": [2, 5, 10],  # Minimum samples required in a leaf node to prevent overfitting
        "max_features": ['auto', 'sqrt', 'log2']  # Number of features to consider for splits
    }

    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(**model_params),
        param_grid=grid_params,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=kfold_params["inner_folds"], shuffle=True, random_state=kfold_params["random_state"]),
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best grid-searched model
    optimized_model = grid_search.best_estimator_
    print(f"\nBest Grid Search Params: {grid_search.best_params_}")

    # Initialise the metrics dictionary to store metrics after predicting the different folds
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    # Use the same folds for kfold cross validation with the optimized model
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        val_pred = optimized_model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

        # Aggregate results for the optimized model
        avg_val_metrics = {
            "accuracy": np.mean(val_metrics["accuracy"]),
            "f1": np.mean(val_metrics["f1"]),
            "precision": np.mean(val_metrics["precision"]),
            "recall": np.mean(val_metrics["recall"])
        }
        std_val_metrics = {
            "accuracy": np.std(val_metrics["accuracy"]),
            "f1": np.std(val_metrics["f1"]),
            "precision": np.std(val_metrics["precision"]),
            "recall": np.std(val_metrics["recall"])
        }
        print("\nAggregated Metrics of Optimized RandomForest Models:")
        print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
        print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
        print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
        print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

        # Save aggregated results into a dataframe
        results_df = pd.DataFrame({
            "model_type": ["Optimized RF"],
            "avg_accuracy": [avg_val_metrics["accuracy"]],
            "avg_f1": [avg_val_metrics["f1"]],
            "avg_precision": [avg_val_metrics["precision"]],
            "avg_recall": [avg_val_metrics["recall"]],
            "std_accuracy": [std_val_metrics["accuracy"]],
            "std_f1": [std_val_metrics["f1"]],
            "std_precision": [std_val_metrics["precision"]],
            "std_recall": [std_val_metrics["recall"]]
        })

        # Append the results_df with the upstream aggegrated metrics 
        agg_val_metrics = pd.concat([upstream_agg_val_metrics, results_df], axis=0, ignore_index=False)

        # Get the accuracy results after checking on each parameters in the grid search
        gr_param_acc_comparison = grid_search.cv_results_

        # Create a DataFrame to view the parameters and accuracies
        gr_param_acc_comparison_df = pd.DataFrame(gr_param_acc_comparison)
        
        # Sort by accuracy
        gr_param_acc_comparison_df = gr_param_acc_comparison_df.sort_values(by='mean_test_score', ascending=False)

        # Return the metrics
        return agg_val_metrics, optimized_model, gr_param_acc_comparison_df
    

def train_lightgbm(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Train a LightGBM classifier using K-Fold cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for LGBMClassifier.
        kfold_params (dict): Parameters for StratifiedKFold.
        
    """
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**model_params)

        model.fit(X_fold_train, y_fold_train)
        
        val_pred = model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

        # Store the model
        models.append(model)

        # Display fold metrics
        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Select the best model based on validation accuracy
    best_unoptimized_model_idx = np.argmax(val_metrics["accuracy"])
    best_unoptimized_model = models[best_unoptimized_model_idx]
    print(f"\nBest Unoptimized Model Fold: {best_unoptimized_model_idx + 1}")

    # Aggregated metrics
    avg_val_metrics = {
        "accuracy": np.mean(val_metrics["accuracy"]),
        "f1": np.mean(val_metrics["f1"]),
        "precision": np.mean(val_metrics["precision"]),
        "recall": np.mean(val_metrics["recall"])
    }
    std_val_metrics = {
        "accuracy": np.std(val_metrics["accuracy"]),
        "f1": np.std(val_metrics["f1"]),
        "precision": np.std(val_metrics["precision"]),
        "recall": np.std(val_metrics["recall"])
    }

    print("\nAggregated Metrics of Unoptimized LightGBM Models:")
    print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
    print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
    print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
    print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

    # Save aggregated results into a dataframe
    results_df = pd.DataFrame({
        "model_type": ["Unoptimized LightBGM"],
        "avg_accuracy": [avg_val_metrics["accuracy"]],
        "avg_f1": [avg_val_metrics["f1"]],
        "avg_precision": [avg_val_metrics["precision"]],
        "avg_recall": [avg_val_metrics["recall"]],
        "std_accuracy": [std_val_metrics["accuracy"]],
        "std_f1": [std_val_metrics["f1"]],
        "std_precision": [std_val_metrics["precision"]],
        "std_recall": [std_val_metrics["recall"]]
    })

    # Append the results_df with the upstream aggegrated metrics 
    agg_val_metrics = pd.concat([upstream_agg_val_metrics, results_df], axis=0, ignore_index=False)

    # Return the metrics and best unoptimized model
    return agg_val_metrics, best_unoptimized_model


def optimize_lightgbm(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize a LightGBM classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for LGBMClassifier.
        kfold_params (dict): Parameters for StratifiedKFold.
        
    """
    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    # Parameters to perform grid search on
    grid_params = {
        "max_depth": [1, 2, 4, 8, 16], # Maximum depth of trees. Higher may overfit
        "num_leaves": [10, 30, 50],  # Number of leaves, higher can improve model but may lead to overfitting
        "n_estimators": [200, 300], # Number of boosting iterations (trees)
        "max_bin": [30, 63, 127], # Number of bins used for discretizing features
        "min_data_in_leaf": [10, 50, 100], # Minimum data in each leaf, higher values reduce overfitting
        "reg_lambda": [0, 0.1, 1, 10], # L2 regularization to reduce overfitting
    }
    grid_search = GridSearchCV(
        estimator=lgb.LGBMClassifier(**model_params),
        param_grid=grid_params,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=kfold_params["inner_folds"], shuffle=True, random_state=kfold_params["random_state"]),
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best grid-searched model
    optimized_model = grid_search.best_estimator_
    print(f"\nBest Grid Search Params: {grid_search.best_params_}")

    # Initialise the metrics dictionary for kfold cross-validation with the optimized model
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    # Use the same folds for kfold cross-validation with the optimized model
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        val_pred = optimized_model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

    # Aggregated metrics for optimized model
    avg_val_metrics = {
        "accuracy": np.mean(val_metrics["accuracy"]),
        "f1": np.mean(val_metrics["f1"]),
        "precision": np.mean(val_metrics["precision"]),
        "recall": np.mean(val_metrics["recall"])
    }
    std_val_metrics = {
        "accuracy": np.std(val_metrics["accuracy"]),
        "f1": np.std(val_metrics["f1"]),
        "precision": np.std(val_metrics["precision"]),
        "recall": np.std(val_metrics["recall"])
    }

    print("\nAggregated Metrics of Optimized LightGBM Models:")
    print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
    print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
    print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
    print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

    # Save aggregated results into a dataframe
    results_df = pd.DataFrame({
        "model_type": ["Optimized LightBGM"],
        "avg_accuracy": [avg_val_metrics["accuracy"]],
        "avg_f1": [avg_val_metrics["f1"]],
        "avg_precision": [avg_val_metrics["precision"]],
        "avg_recall": [avg_val_metrics["recall"]],
        "std_accuracy": [std_val_metrics["accuracy"]],
        "std_f1": [std_val_metrics["f1"]],
        "std_precision": [std_val_metrics["precision"]],
        "std_recall": [std_val_metrics["recall"]]
    })

    # Append the results with upstream aggregated metrics
    agg_val_metrics = pd.concat([upstream_agg_val_metrics, results_df], axis=0, ignore_index=False)

    # Get the accuracy results after checking on each parameter in the grid search
    gr_param_acc_comparison = grid_search.cv_results_

    # Create a DataFrame to view the parameters and accuracies
    gr_param_acc_comparison_df = pd.DataFrame(gr_param_acc_comparison)
    gr_param_acc_comparison_df = gr_param_acc_comparison_df.sort_values(by='mean_test_score', ascending=False)

    # Return the results dataframe, the optimized model, and the grid search details
    return agg_val_metrics, optimized_model, gr_param_acc_comparison_df


def train_svm(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Train an SVM classifier using K-Fold cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for SVC.
        kfold_params (dict): Parameters for StratifiedKFold.
        
    """
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = SVC(**model_params)
        model.fit(X_fold_train, y_fold_train)
        
        val_pred = model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

        # Store the model along with the associated indices for train-val split
        models.append(model)

        # Display fold metrics
        print(f"Fold {fold + 1}:")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Select the best model with highest validation accuracy to represent the unoptimized model
    best_unoptimized_model_idx = np.argmax(val_metrics["accuracy"])
    best_unoptimized_model = models[best_unoptimized_model_idx]
    print(f"\nBest Unoptimized Model Fold: {best_unoptimized_model_idx + 1}")

    # Display the aggregated metrics of the unoptimized models
    avg_val_metrics = {
        "accuracy": np.mean(val_metrics["accuracy"]),
        "f1": np.mean(val_metrics["f1"]),
        "precision": np.mean(val_metrics["precision"]),
        "recall": np.mean(val_metrics["recall"])
    }
    std_val_metrics = {
        "accuracy": np.std(val_metrics["accuracy"]),
        "f1": np.std(val_metrics["f1"]),
        "precision": np.std(val_metrics["precision"]),
        "recall": np.std(val_metrics["recall"])
    }
    print("\nAggregated Metrics of Unoptimized SVM Models:")
    print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
    print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
    print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
    print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

    # Save aggregated results into a dataframe for passing downstream
    results_df = pd.DataFrame({
        "model_type": ["Unoptimized SVM"],
        "avg_accuracy": [avg_val_metrics["accuracy"]],
        "avg_f1": [avg_val_metrics["f1"]],
        "avg_precision": [avg_val_metrics["precision"]],
        "avg_recall": [avg_val_metrics["recall"]],
        "std_accuracy": [std_val_metrics["accuracy"]],
        "std_f1": [std_val_metrics["f1"]],
        "std_precision": [std_val_metrics["precision"]],
        "std_recall": [std_val_metrics["recall"]]
    })

    # Append the results_df with the upstream aggegrated metrics 
    agg_val_metrics = pd.concat([upstream_agg_val_metrics, results_df], axis=0, ignore_index=False)

    # Return the metrics and best unoptimized model
    return agg_val_metrics, best_unoptimized_model


def optimize_svm(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize an SVM classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for SVC.
        kfold_params (dict): Parameters for StratifiedKFold.
        
    """
    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    # Parameters to perform the grid search on
    grid_params = {
        "C": [1e-2, 1e0, 1e2],
        "kernel": ['linear', 'rbf', 'poly'],
          "tol": [1e-3, 1e-4, 1e-5],
    }
    grid_search = GridSearchCV(
        estimator=SVC(**model_params),
        param_grid=grid_params,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=kfold_params["inner_folds"], shuffle=True, random_state=kfold_params["random_state"]),
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get the best grid-searched model
    optimized_model = grid_search.best_estimator_
    print(f"\nBest Grid Search Params: {grid_search.best_params_}")

    # Initialise the metrics dictionary to store metrics after predicting the different folds
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    # Use the same folds for kfold cross validation with the optimized model
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        val_pred = optimized_model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="weighted")
        precision = precision_score(y_fold_val, val_pred, average="weighted")
        recall = recall_score(y_fold_val, val_pred, average="weighted")

        # Store the metrics
        val_metrics["accuracy"].append(acc)
        val_metrics["f1"].append(f1)
        val_metrics["precision"].append(precision)
        val_metrics["recall"].append(recall)

        # Aggregate results for the optimized model
        avg_val_metrics = {
            "accuracy": np.mean(val_metrics["accuracy"]),
            "f1": np.mean(val_metrics["f1"]),
            "precision": np.mean(val_metrics["precision"]),
            "recall": np.mean(val_metrics["recall"])
        }
        std_val_metrics = {
            "accuracy": np.std(val_metrics["accuracy"]),
            "f1": np.std(val_metrics["f1"]),
            "precision": np.std(val_metrics["precision"]),
            "recall": np.std(val_metrics["recall"])
        }
        print("\nAggregated Metrics of Optimized SVM Models:")
        print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
        print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
        print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
        print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

        # Save aggregated results into a dataframe
        results_df = pd.DataFrame({
            "model_type": ["Optimized SVM"],
            "avg_accuracy": [avg_val_metrics["accuracy"]],
            "avg_f1": [avg_val_metrics["f1"]],
            "avg_precision": [avg_val_metrics["precision"]],
            "avg_recall": [avg_val_metrics["recall"]],
            "std_accuracy": [std_val_metrics["accuracy"]],
            "std_f1": [std_val_metrics["f1"]],
            "std_precision": [std_val_metrics["precision"]],
            "std_recall": [std_val_metrics["recall"]]
        })

        # Append the results_df with the upstream aggregated metrics 
        agg_val_metrics = pd.concat([upstream_agg_val_metrics, results_df], axis=0, ignore_index=False)

        # Get the accuracy results after checking on each parameter in the grid search
        gr_param_acc_comparison = grid_search.cv_results_

        # Create a DataFrame to view the parameters and accuracies
        gr_param_acc_comparison_df = pd.DataFrame(gr_param_acc_comparison)
        
        # Sort by accuracy
        gr_param_acc_comparison_df = gr_param_acc_comparison_df.sort_values(by='mean_test_score', ascending=False)

        # Return the metrics
        return agg_val_metrics, optimized_model, gr_param_acc_comparison_df


def test_models(X_test, y_test, 
                unoptimized_rf,
                optimised_rf,
                unoptimised_lightbgm,
                optmised_lightbgm,
                unoptimised_svm,
                optimised_svm):
    
    # Identify the models and put them into a list for identification
    models = [unoptimized_rf, optimised_rf, unoptimised_lightbgm, optmised_lightbgm, unoptimised_svm, optimised_svm]
    model_names = ['Unoptimized RF', 'Optimized RF', 'Unoptimized LightBGM', 'Optimized LightBGM', 'Unoptimized SVM', 'Optimized SVM',]

    # Initialize a list to store the test prediction results
    test_results = []

    for i, model in enumerate(models):
        # Identify the name of the model
        model_name = model_names[i]

        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        # Append results
        test_results.append({
            "Model": f"{model_name}",
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall
        })

    # Convert test results to DataFrame
    test_results_df = pd.DataFrame(test_results)
    return test_results_df


def make_inference(
        df,
        feature_encoded_testing_df,
        unoptimized_rf_model,
        optimized_rf_model,
        unoptimized_lightbgm_model,
        optimized_lightbgm_model,
        unoptimized_svm_model,
        optimized_svm_model
        ):
    
    # Create a copy of the original dataframe for each model's predictions
    inference_unoptimized_rf_df = df.copy()
    inference_optimized_rf_df = df.copy()
    inference_unoptimized_lightbgm_df = df.copy()
    inference_optimized_lightbgm_df = df.copy()
    inference_unoptimized_svm_df = df.copy()
    inference_optimized_svm_df = df.copy()

    # Make predictions using each model
    inference_unoptimized_rf_df['survived_prediction'] = unoptimized_rf_model.predict(feature_encoded_testing_df)
    inference_optimized_rf_df['survived_prediction'] = optimized_rf_model.predict(feature_encoded_testing_df)
    inference_unoptimized_lightbgm_df['survived_prediction'] = unoptimized_lightbgm_model.predict(feature_encoded_testing_df)
    inference_optimized_lightbgm_df['survived_prediction'] = optimized_lightbgm_model.predict(feature_encoded_testing_df)
    inference_unoptimized_svm_df['survived_prediction'] = unoptimized_svm_model.predict(feature_encoded_testing_df)
    inference_optimized_svm_df['survived_prediction'] = optimized_svm_model.predict(feature_encoded_testing_df)
    
    return {
        "inference_unoptimized_rf_df": inference_unoptimized_rf_df,
        "inference_optimized_rf_df": inference_optimized_rf_df,
        "inference_unoptimized_lightbgm_df": inference_unoptimized_lightbgm_df,
        "inference_optimized_lightbgm_df": inference_optimized_lightbgm_df,
        "inference_unoptimized_svm_df": inference_unoptimized_svm_df,
        "inference_optimized_svm_df": inference_optimized_svm_df
    }