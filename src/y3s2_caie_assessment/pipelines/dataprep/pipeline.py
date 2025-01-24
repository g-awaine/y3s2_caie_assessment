from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the data preparation pipeline.
    """
    return Pipeline(
        [
            node(
                func=drop_missing_rows,
                inputs=["train_df", "params:missing_entries_columns"],
                outputs="train_df_dropped_rows",
                name="drop_missing_rows_train",
            ),

        ]

)
