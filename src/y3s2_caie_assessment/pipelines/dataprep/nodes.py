import re
import difflib
import pandas as pd
from datetime import datetime
from typing import List, Callable, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def drop_rows_conditional(df: pd.DataFrame, conditions: dict) -> pd.DataFrame:
    """
    Removes rows from the DataFrame that do not match the specified conditions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        conditions (dict): A dictionary where keys are column names and values are the conditions.

    Returns:
        pd.DataFrame: The DataFrame with rows matching the conditions removed.
    """
    try:
        # Initialize the mask for rows to keep
        mask = pd.Series(True, index=df.index)
        
        for column, condition in conditions.items():
            # Apply the condition and remove rows which do not meet the condition
            mask &= df[column].apply(condition)
        
        # Return the filtered DataFrame
        return df[mask]
    
    except Exception as e:
        # Showcase the error
        print(f"An unexpected error occurred: {e}")
        return df

def drop_rows_missing_values(df: pd.DataFrame, subset: List[str], how: str) -> pd.DataFrame:
    """
    Removes rows from the DataFrame that have missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        subset (List[str]): The columns to search for missing values.
        how (str): Determines whether to drop rows with any missing or all missing values.

    Returns:
        pd.DataFrame: The DataFrame with the missing values removed from the specified columns.
    """
    try:
        # Check if columns exist in the DataFrame
        missing_columns = [col for col in subset if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in the DataFrame: {', '.join(missing_columns)}")

        # Drop rows where the specified columns/fields have null entries
        return df.dropna(subset=subset, how=how)
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return df


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Drops the specified columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns_to_drop (List[str]): List of the specified columns to be removed from the Dataframe.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns removed.
    """
    try:
        # Check if all specified columns exist in the DataFrame
        missing_columns = [col for col in columns_to_drop if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in the DataFrame: {', '.join(missing_columns)}")
        
        # Drop the specified columns from the DataFrame
        return df.drop(columns=columns_to_drop)
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return df 

def aggregate_by_column(df: pd.DataFrame, column: str, agg: dict) -> pd.DataFrame:
    """
    Aggregates the dataset by the specified column according to the specified aggregation functions.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to group by.
        agg (str): The dictionary to aggregate the other columns by.

    Returns:
        pd.DataFrame: The aggregated DataFrame.
    """
    try:
        # Check if the column exist in the DataFrame
        is_column_missing = True if column not in df.columns else False
        if is_column_missing:
            raise ValueError(f"Column not found in the DataFrame")
        
        # Aggregate the input DataFrame by the specified column
        aggregated_df = df.groupby(column).agg(agg)

        # Return the aggregated Dataframe
        return aggregated_df
    
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return df

    except Exception as e:
        # Show the error
        print(f"An unexpected error occurred: {e}")
        return df
    
def replace_diacritics(text: str) -> str:
    """
    Replaces diacritics with standard alphabets.

    Args:
        text (str): The input string which may contain diacritic.

    Returns:
        str: Outputs a string where the diacritics is replaced with standard alphabet letters.
    """
    # Initialise the regular expressions for diacritics conversion
    replacements = {
        r'[ãââàáä]': 'a',
        r'[íîì]': 'i',
        r'[úûùü]': 'u',
        r'[éêèë]': 'e',
        r'[óõôòö]': 'o',
        r'[ç]': 'c'
    }

    # Checks if the text is a string
    if isinstance(text, str):
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        return text
    else:
        raise ValueError("Text must be a string")

def identify_closest_text(text: str, text_list: List[str]) -> str:
    """
    Identify closest text from the list of text.

    Args:
        text (str): The input string to be compared.
        text_list (List[str]): The list of text to be compared with the text to identify the closest one.

    Returns:
        str: The closest text in the list.
    """
    try:
        # Identify the closest text
        closest_text = difflib.get_close_matches(text, text_list)[0]
        return closest_text
    
    except Exception as e:
        # Show error
        print(f"An unexpected error occurred: {e}")

def standardize_cities(df: pd.DataFrame, city_column: str, city_list: List[str]) -> pd.DataFrame:
    """
    Standardizes the names of the cities in the city column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        city_column (str): The column containing the city names.
        city_list (List[str]): The list containing the preprocessed true names of all the cities in Brazil without encoding errors or diacritics e.g. sao paulo.

    Returns:
        pd.DataFrame: The DataFrame with standardised names for the cities.
    """
    try:
        # Check if the column exist in the DataFrame
        is_column_missing = True if city_column not in df.columns else False
        if is_column_missing:
            raise ValueError(f"Column not found in the DataFrame")
        
        # Replaces diacritics with plain alphabets
        df[city_column] = df[city_column].apply(replace_diacritics)

        # Identifies the closest city match
        df[city_column] = df[city_column].apply(lambda x: identify_closest_text(x, city_list))
        
        return df
    
    except ValueError as ve:
        # Show error
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        # Show error
        print(f"An unexpected error occurred: {e}")
        return df


def mice_impute_entries(df: pd.DataFrame, imputation_parameters: dict) -> pd.DataFrame:
    """
    Imputes entries in the DataFrame using MICE.
    
    Args:
        df (pd.DataFrame): The dataset with missing entries.
        imputation_parameters (dict): A dictionary that contains the parameters for imputing the missing age entries
            - significant_numerical_columns (List[str]): The numerical columns which are significant for the MICE imputer to fit by.
            - significant_categorical_columns (List[str]): The categorical columns which are significant for the MICE imputer to fit by.
            - max_iter (int, optional): The maximum number of imputation iterations. Default is 10.
            - random_state (int, optional): The seed used by the MICE imputer. Default is 0.
    
    Returns:
        pd.DataFrame: The dataset with imputed entries.
    """
    try:
        # Identify the numerical and categorical columns from the significant columns
        numerical_cols = imputation_parameters.get('significant_numerical_columns', [])
        categorical_cols = imputation_parameters.get('significant_categorical_columns', [])

        # Get the max iteration and random state for the MICE imputer
        max_iter = imputation_parameters.get('max_iter', 10)
        random_state = imputation_parameters.get('random_state', 42)

        # Define a list containing all the significant columns above
        significant_cols = numerical_cols + categorical_cols

        # Check if these columns exist in the DataFrame
        missing_columns = [col for col in significant_cols if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns not found in the DataFrame: {', '.join(missing_columns)}")

        # Identify the columns which are not used in the imputation
        columns_not_used = [col for col in df.columns if col not in significant_cols]

        # Instantiate the IterativeImputer (MICE imputer), OneHotEncoder and StandardScaler
        mice_imputer = IterativeImputer(max_iter=max_iter, 
                                        random_state=random_state,
                                        min_value=0.17,
                                        max_value=80,
                                        initial_strategy='median')
        
        ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = StandardScaler()

        # Obtain the subset of the dataset
        imputing_df = df[significant_cols]

        # Scale the numerical fields with standard scaling
        if numerical_cols:
            scaled_numerical_data = scaler.fit_transform(imputing_df[numerical_cols])
            scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_cols, index=imputing_df.index)
        else:
            scaled_numerical_df = pd.DataFrame()

        # One hot encode the categorical fields
        if categorical_cols:
            encoded_categorical_data = ohe_encoder.fit_transform(imputing_df[categorical_cols])
            encoded_categorical_df = pd.DataFrame(
                encoded_categorical_data,
                columns=ohe_encoder.get_feature_names_out(categorical_cols),
                index=imputing_df.index
            )
        else:
            encoded_categorical_df = pd.DataFrame()

        # Combine the processed numerical and categorical data
        processed_df = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

        # Apply MICE imputation and convert the data to a dataframe
        imputed_data = mice_imputer.fit_transform(processed_df)
        imputed_df = pd.DataFrame(imputed_data, columns=processed_df.columns, index=processed_df.index)

        # Restore the numerical scaling to the original
        if numerical_cols:
            restored_numerical_data = scaler.inverse_transform(imputed_df[numerical_cols])
            restored_numerical_df = pd.DataFrame(
                restored_numerical_data, columns=numerical_cols, index=imputed_df.index
            )
        else:
            restored_numerical_df = pd.DataFrame()

        # Restore categorical columns to their original categories
        if categorical_cols:
            restored_categorical_data = ohe_encoder.inverse_transform(imputed_df[ohe_encoder.get_feature_names_out(categorical_cols)])
            restored_categorical_df = pd.DataFrame(
                restored_categorical_data, columns=categorical_cols, index=imputed_df.index
            )
        else:
            restored_categorical_df = pd.DataFrame()

        # Combine the restored numerical and categorical columns
        restored_df = pd.concat([restored_numerical_df, restored_categorical_df], axis=1)

        # Combine the restored dataframe with the other columns that were not used
        if columns_not_used:
            restored_df[columns_not_used] = df[columns_not_used]

        return restored_df
    
    except ValueError as ve:
        # Show the error
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        # Show the error
        print(f"An unexpected error occurred: {e}")
        return df


def simple_impute_entries(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """
    Imputes missing entries in the DataFrame using a specified value.
    
    Args:
        df (pd.DataFrame): The input dataset with missing entries.
        column (str): The column to impute.
        value (str): The value to impute into the empty cell.
    
    Returns:
        pd.DataFrame: The dataset with imputed entries.
    """
    try:
        # Check if the column exist in the DataFrame
        is_column_missing = True if column not in df.columns else False
        if is_column_missing:
            raise ValueError(f"Column not found in the DataFrame")
        
        # Impute the empty cells in the column with the specified value
        df[column] = df[column].fillna(value)
        
    except ValueError as ve:
        # Show error
        print(f"ValueError: {ve}")
        return df
    
    except Exception as e:
        # Show error
        print(f"An unexpected error occurred: {e}")
        return df


def date_difference(row: pd.Series, params: dict) -> int:
    """
    Calculate the difference in days between two dates.

    Parameters:
        row (pd.Series): The row from the dataset.
        mapping (dict): A dictionary containing:
            - 'date1': Column name for the first date.
            - 'date2': Column name for the second date.
            - 'date1_format': Format of the first date.
            - 'date2_format': Format of the second date.

    Returns:
        int: The difference in days between the two dates.
    """
    try:
        # Identify the 2 dates
        date1 = row[params['date1']]
        date2 = row[params['date2']]

        # Check if there are any missing dates
        if pd.isna(date1) or pd.isna(date2):
            return None # Returns None when its not delivered yet
        
        # Parse the dates
        d1 = datetime.strptime(date1, params['date1_format'])
        d2 = datetime.strptime(date2, params['date2_format'])

        # Calculate the difference
        day_difference = (d2 - d1).days
        return day_difference
    
    except ValueError as ve:
        raise ValueError(f"Invalid date or format: {ve}")
    
def feature_engineering(df: pd.DataFrame, new_feature: str, function: Callable, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Create a new feature based on the features in the row.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        new_feature (str): The column name of the new feature to be created.
        function (Callable): The function to be applied in order to create the feature.
        mapping (List[str]): Mapping column names to the parameter names used by the function. Provides other arguments for the function.

    Returns:
        pd.DataFrame: The DataFrame with new features.
    """
    try:
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
    