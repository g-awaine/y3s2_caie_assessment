from kedro.pipeline import Pipeline, node
from .nodes import *


def create_pipeline(**kwargs) -> Pipeline:
    """
    Creates the data preparation pipeline.
    """
    return Pipeline(
        [
            node(
                func=drop_duplicate,
                inputs=["geolocation_df"],
                outputs="dropped_dupes_geolocation_df",
                name="remove_duplicates_geolocation"
            ),
            node(
                func=drop_rows_lat_lng,
                inputs=["dropped_dupes_geolocation_df", "params:clean_lat_lng_conditions"],
                outputs="clean_lat_lng_geolocation_df",
                name="clean_lat_lng_geolocation"
            ),
            node(
                func=aggregate_by_column,
                inputs=["clean_lat_lng_geolocation_df", "params:geolocation_groupby_column", "param:geolocation_agg_mapping"],
                outputs="agg_geolocation_df",
                name="aggregate_geolocation"
            ),
            node(
                func=cross_reference_cities,
                inputs=["agg_geolocation_df", "customers_df", "param:geolocation_customers_city_map"],
                outputs="cross_referenced_geolocation_df",
                name="cross_reference_cities_geolocation"
            ),
            node(
                func=standardize_cities,
                inputs=["cross_referenced_geolocation_df", "param:geolocation_city_column"],
                outputs="cleaned_geolocation_df",
                name="standardised_geolocation_city_spelling"
            ),
            node(
                func=drop_rows_missing_values,
                inputs=["products_df", "param: missing_products_subset", "all"],
                outputs="dropped_missing_values_products_df",
                name="drop_missing_values_products"
            ),
            node(
                func=mice_impute_entries,
                inputs=["products_df", "param: mice_imputation_parameters"],
                outputs="imputed_products_df",
                name="impute_products"
            ),
            node(
                func=drop_columns,
                inputs=["orders_df", "param: orders_columns_to_drop"],
                outputs="dropped_columns_orders_df",
                name="drop_columns_products"
            ),
            node(
                func=drop_erroneous_orders,
                inputs=["dropped_columns_orders_df"],
                outputs="dropped_orders_products_df",
                name="drop_erroneous_orders_products"
            ),
            node(
                func=feature_engineering,
                inputs=["dropped_orders_products_df",
                        "param:new_feature_orders",
                        date_difference,
                        "param:feature_engineering_orders_map"
                        ],
                outputs="feature_engineered_orders_df",
                name="feature_engineering_orders"
            ),
            node(
                func=drop_columns,
                inputs=["reviews_df", "param: orders_columns_to_drop"],
                outputs="dropped_columns_reviews_df",
                name="drop_columns_reviews"
            ),
            node(
                func=cross_reference_cities,
                inputs=["sellers_df", "customers_df", "param:sellers_customers_city_map"],
                outputs="cross_referenced_sellers_df",
                name="cross_reference_cities_sellers"
            ),
            node(
                func=standardize_cities,
                inputs=["cross_referenced_sellers_df", "param:sellers_city_column"],
                outputs="cleaned_sellers_df",
                name="standardised_sellers_city_spelling"
            )
        ]
)
