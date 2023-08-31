# Author: CHEN Ee Heng
# Date: 31.08.2023

import pandas as pd


def cell_list_to_columns(df: pd.DataFrame,
                         column: str,
                         length: int
                         ) -> pd.DataFrame:
    """Put each element of a list in a cell to a separate column.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column that has list in the cell.
        length (int): Length of list.

    Returns:
        pd.DataFrame: Dataframe with 1 element per cell.
    """
    new_columns = [f'{column}_{i}' for i in range(length)]
    return pd.DataFrame(df[column].to_list(),
                        columns=new_columns,
                        index=df.index)


def cell_list_to_columns_and_max(df: pd.DataFrame,
                                 column: str,
                                 length: int
                                 ) -> pd.DataFrame:
    """Put each element of a list in a cell to a separate column and calculate 
    the max() based on 'map_id'.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column that has list in the cell.
        length (int): Length of list.

    Returns:
        pd.DataFrame: Dataframe with 1 element per cell.
    """
    new_columns = [f'{column}_{i}' for i in range(length)]
    df_tmp = cell_list_to_columns(df, column, length)
    df_tmp = pd.concat([df['map_id'], df_tmp], axis=1)
    df_tmp = df_tmp.groupby('map_id')[new_columns].transform('max')
    return df_tmp


def pad_zeros(x: list, n: int) -> list:
    return (x + [0] * (n - len(x)))[:n]


def pad_cell_list_to_512(df: pd.DataFrame, column: int) -> pd.DataFrame:
    df[f'{column}_pad512'] = df[f'{column}'].apply(lambda x: pad_zeros(x, 512))
    return df
