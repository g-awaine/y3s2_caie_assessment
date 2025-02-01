"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=standardisation_and_encoding,
                inputs=["final_modelling_dataset",
                        "params:modelling_dataset.label_col",
                        "params:modelling_dataset.numerical_cols", 
                        "params:modelling_dataset.categorical_cols"],
                outputs="encoded_modelling_dataset",
                name="feature_standardisation_and_encoding",
            ),
            node(
                func=train_test_split,
                inputs=["encoded_modelling_dataset",
                        "params:modelling_dataset.label_col",
                        "params:split_parameters", 
                        "params:random_state"],
                outputs=["X_train", "y_train", "X_test", "y_test"],
                name="train_test_split",
            ),
            node(
                func=train_random_forest,
                inputs=["X_train", "y_train", "params:rf_params", "params:kfold_params"],
                outputs=["agg_val_metrics_0", "unoptimized_rf_model"],
                name="train_random_forest",
            ),
            node(
                func=optimize_random_forest,
                inputs=["X_train", "y_train", "params:rf_params", "params:kfold_params", "params:grid_params.rf", "agg_val_metrics_0"],
                outputs=["agg_val_metrics_1", "optimized_rf_model", "gr_rf_param_acc_comparison"],
                name="optimize_random_forest",
            ),
            node(
                func=train_lightgbm,
                inputs=["X_train", "y_train", "params:lightbgm_params", "params:kfold_params", "agg_val_metrics_1"],
                outputs=["agg_val_metrics_2", "unoptimized_lightbgm_model"],
                name="train_lightgbm",
            ),
            node(
                func=optimize_lightgbm,
                inputs=["X_train", "y_train", "params:lightbgm_params", "params:kfold_params", "params:grid_params.lightgbm", "agg_val_metrics_2"],
                outputs=["agg_val_metrics_3", "optimized_lightbgm_model", "gr_lightbgm_param_acc_comparison"],
                name="optimize_lightgbm",
            ),
            node(
                func=train_lr,
                inputs=["X_train", "y_train", "params:lr_params", "params:kfold_params", "agg_val_metrics_3"],
                outputs=["agg_val_metrics_4", "unoptimized_lr_model"],
                name="train_svm",
            ),
            node(
                func=optimize_lr,
                inputs=["X_train", "y_train", "params:lr_params", "params:kfold_params", "params:grid_params.lr", "agg_val_metrics_4"],
                outputs=["agg_val_metrics_5", "optimized_lr_model", "gr_lr_param_acc_comparison"],
                name="optimize_lr",
            ),
            node(
                func=test_models,
                inputs=["X_test", "y_test",
                        "unoptimized_rf_model", "optimized_rf_model", 
                        "unoptimized_lightbgm_model", "optimized_lightbgm_model", 
                        "unoptimized_lr_model", "optimized_lr_model"],
                outputs="test_results",
                name="test_models",
            ),
            node(
                func=make_prediction,
                inputs=["final_modelling_dataset", "encoded_modelling_dataset",
                        "unoptimized_rf_model", "optimized_rf_model", 
                        "unoptimized_lightbgm_model", "optimized_lightbgm_model", 
                        "unoptimized_lr_model", "optimized_lr_model"],
                outputs={
                    "inference_unoptimized_rf_df": "inference_unoptimized_rf_df",
                    "inference_optimized_rf_df": "inference_optimized_rf_df",
                    "inference_unoptimized_lightbgm_df": "inference_unoptimized_lightbgm_df",
                    "inference_optimized_lightbgm_df": "inference_optimized_lightbgm_df",
                    "inference_unoptimized_lr_df": "inference_unoptimized_lr_df",
                    "inference_optimized_lr_df": "inference_optimized_lr_df"
                },
                name="make_prediction"
            )
        ]
    )
