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
                func=feature_engineering,
                inputs=["dropped_orders",
                        "params:orders_modelling_params.features.new_feature",
                        "params:orders_modelling_params.features.function_name",
                        "params:orders_modelling_params.date_column_mapping"
                        ],
                outputs="orders_feature_engineered",
                name="feature_engineering_orders"
            ),

        ]
)
