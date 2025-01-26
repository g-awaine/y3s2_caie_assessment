import pandas as pd
import datetime as datetime
from typing import List, Dict
import numpy as np
import lightgbm as lgb

from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline



def standardisation_and_encoding(df: pd.DataFrame, label_col:str, numerical_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
    """
    Encodes the categorical data into One Hot Encoded vectors and standardises the scaling of numerical data
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        label_col (str): The column containing the label of the dataset.
        numerical_cols (List[str]): The list of numerical features in the DataFrame.
        categorical_cols (List[str]): The list of categorical features in the DataFrame.

    Returns:
        feature_encoded_df (pd.DataFrame): The DataFrame after feature encoding and standardisation.
    """
    try:
        # Combine the numerical and categorical column lists
        columns = numerical_cols + categorical_cols

        # Check if all columns are included in the standardisation and encoding
        if set(columns) != set(df.columns.drop(label_col)):
            raise ValueError(f"Not all columns were included")

        # Initialize encoders and scalers
        label_encoder = LabelEncoder()
        scaler = StandardScaler()

        # Process categorical columns
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Process numerical columns
        scaled_numerical_data = scaler.fit_transform(df[numerical_cols])
        scaled_numerical_df = pd.DataFrame(
            scaled_numerical_data,
            columns=numerical_cols,
            index=df.index
        )

        # Combine the processed data 
        feature_encoded_df = pd.concat([scaled_numerical_df, df[categorical_cols], df[label_col]], axis=1)
        return feature_encoded_df
    
    except ValueError as ve:
        # Show error
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        # Show the error
        print(f"An unexpected error occurred: {e}")
        return df


def train_test_split(df: pd.DataFrame, label_column: str, split_parameters: dict, random_state: int):
    """
    Uses stratified shuffle split to split the dataframe into evenly distributed train and test datasets for modelling

    Args:
        df (pd.DataFrame): The input standardised and encoded dataframe to be split for modelling.
        label_column (str): The column that contains the labels.
        split_parameters (dict): The split parameters for stratified shuffle split.
        random_state (int): The random state value to be used.

    Returns:
        X_train: The features used for training
        y_train: The labels of the dataset used for training
        X_test: The features used for testing
        y_test: The labels of the dataset used for testing
    """
    X = df.drop(label_column, axis=1)
    y = df[label_column]
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
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Apply random under-sampling to the training data
        X_fold_train, y_fold_train = rus.fit_resample(X_fold_train, y_fold_train)
        
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


def optimize_random_forest(X_train, y_train, model_params, kfold_params, grid_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize a Random Forest classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for RandomForestClassifier.
        kfold_params (dict): Parameters for StratifiedKFold.
        grid_params (dict): Parameters for the grid search
        upstream_agg_val_metrics (pd.DataFrame): The dataframe to add the validation metrics to.
    """
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    # Create the pipeline to ensure that RUS is applied inside the grid search
    pipeline = Pipeline([
        ('rus', rus),
        ('rf', RandomForestClassifier(**model_params))
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        scoring="recall",
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
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Apply random under-sampling to the training data
        X_fold_train, y_fold_train = rus.fit_resample(X_fold_train, y_fold_train)

        
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


def optimize_lightgbm(X_train, y_train, model_params, kfold_params, grid_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize a LightGBM classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for LGBMClassifier.
        kfold_params (dict): Parameters for StratifiedKFold.
        grid_params (dict): Parameters for the grid search
        upstream_agg_val_metrics (pd.DataFrame): The dataframe to add the validation metrics to.
        
    """
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    
    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])

    # Create the pipeline to ensure that RUS is applied inside the grid search
    pipeline = Pipeline([
        ('rus', rus),
        ('rf', lgb.LGBMClassifier(**model_params))
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        scoring="recall",
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


def train_lr(X_train, y_train, model_params, kfold_params, upstream_agg_val_metrics):
    """
    Train an Logistic Regression classifier using K-Fold cross-validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for LR.
        kfold_params (dict): Parameters for StratifiedKFold.
        
    """
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)

    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    models = []
    val_metrics = {"accuracy": [], "f1": [], "precision": [], "recall": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Apply random under-sampling to the training data
        X_fold_train, y_fold_train = rus.fit_resample(X_fold_train, y_fold_train)
        
        model = LogisticRegression(**model_params)
        model.fit(X_fold_train, y_fold_train)
        
        val_pred = model.predict(X_fold_val)

        # Calculate the metrics for the validation
        acc = accuracy_score(y_fold_val, val_pred)
        f1 = f1_score(y_fold_val, val_pred, average="macro")
        precision = precision_score(y_fold_val, val_pred, average="macro")
        recall = recall_score(y_fold_val, val_pred, average="macro")

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
    print("\nAggregated Metrics of Unoptimized LR Models:")
    print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
    print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
    print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
    print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

    # Save aggregated results into a dataframe for passing downstream
    results_df = pd.DataFrame({
        "model_type": ["Unoptimized LR"],
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


def optimize_lr(X_train, y_train, model_params, kfold_params, grid_params, upstream_agg_val_metrics):
    """
    Perform grid search to optimize an LR classifier and return the best model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        model_params (dict): Parameters for SVC.
        kfold_params (dict): Parameters for StratifiedKFold.
        grid_params (dict): Parameters for the grid search
        upstream_agg_val_metrics (pd.DataFrame): The dataframe to add the validation metrics to.
        
    """
    # Initialize RandomUnderSampler
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    
    # Initialise the Kfold Splitter
    skf = StratifiedKFold(n_splits=kfold_params['outer_folds'], 
                          shuffle=kfold_params['shuffle'], 
                          random_state=kfold_params['random_state'])
    
    # Create the pipeline to ensure that RUS is applied inside the grid search
    pipeline = Pipeline([
        ('rus', rus),
        ('rf', LogisticRegression(**model_params))
    ])

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid_params,
        scoring="recall",
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
        f1 = f1_score(y_fold_val, val_pred, average="macro")
        precision = precision_score(y_fold_val, val_pred, average="macro")
        recall = recall_score(y_fold_val, val_pred, average="macro")

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
        print("\nAggregated Metrics of Optimized LR Models:")
        print(f"Avg Accuracy: {avg_val_metrics['accuracy']:.4f}")
        print(f"Avg F1: {avg_val_metrics['f1']:.4f}")
        print(f"Avg Precision: {avg_val_metrics['precision']:.4f}")
        print(f"Avg Recall: {avg_val_metrics['recall']:.4f}")

        # Save aggregated results into a dataframe
        results_df = pd.DataFrame({
            "model_type": ["Optimized LR"],
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
                unoptimised_lr,
                optimised_lr):
    """
    Tests the performance of various models on the test data and returns the performance metrics.

    Args:
        X_test (pd.DataFrame): The feature data for testing.
        y_test (pd.Series): The true labels for the test data.
        unoptimized_rf: The unoptimized random forest model.
        optimized_rf: The optimized random forest model.
        unoptimized_lightbgm: The unoptimized LightGBM model.
        optimized_lightbgm: The optimized LightGBM model.
        unoptimised_lr: The unoptimized LR model.
        optimised_lr: The optimized LR model.

    Returns:
        pd.DataFrame: A DataFrame containing the accuracy, F1 score, precision, recall for each model.
    """

    # Identify the models and put them into a list for identification
    models = [unoptimized_rf, optimised_rf, 
              unoptimised_lightbgm, optmised_lightbgm, 
              unoptimised_lr, optimised_lr
    ]
    model_names = [
        'Unoptimized RF', 'Optimized RF', 
        'Unoptimized LightBGM', 'Optimized LightBGM', 
        'Unoptimized LR', 'Optimized LR',
    ]

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

        # Print the classification report
        print(f"\nClassification report for {model_name}:\n")
        print(classification_report(y_test, y_pred))

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


def make_prediction(
        df:pd.DataFrame,
        encoded_df:pd.DataFrame,
        unoptimized_rf_model,
        optimized_rf_model,
        unoptimized_lightbgm_model,
        optimized_lightbgm_model,
        unoptimized_lr_model,
        optimized_lr_model
        ):
    
    # Create a copy of the original dataframe for each model's predictions
    inference_unoptimized_rf_df = df.copy()
    inference_optimized_rf_df = df.copy()
    inference_unoptimized_lightbgm_df = df.copy()
    inference_optimized_lightbgm_df = df.copy()
    inference_unoptimized_lr_df = df.copy()
    inference_optimized_lr_df = df.copy()

    # Make predictions using each model
    inference_unoptimized_rf_df['survived_prediction'] = unoptimized_rf_model.predict(encoded_df)
    inference_optimized_rf_df['survived_prediction'] = optimized_rf_model.predict(encoded_df)
    inference_unoptimized_lightbgm_df['survived_prediction'] = unoptimized_lightbgm_model.predict(encoded_df)
    inference_optimized_lightbgm_df['survived_prediction'] = optimized_lightbgm_model.predict(encoded_df)
    inference_unoptimized_lr_df['survived_prediction'] = unoptimized_lr_model.predict(encoded_df)
    inference_optimized_lr_df['survived_prediction'] = optimized_lr_model.predict(encoded_df)
    
    return {
        "inference_unoptimized_rf_df": inference_unoptimized_rf_df,
        "inference_optimized_rf_df": inference_optimized_rf_df,
        "inference_unoptimized_lightbgm_df": inference_unoptimized_lightbgm_df,
        "inference_optimized_lightbgm_df": inference_optimized_lightbgm_df,
        "inference_unoptimized_lr_df": inference_unoptimized_lr_df,
        "inference_optimized_lr_df": inference_optimized_lr_df
    }