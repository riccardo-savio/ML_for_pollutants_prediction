from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame


def get_percentage_missing_days(df: pd.DataFrame) -> float:
    """
    Calculate the percentage of missing days in a time series DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The time series DataFrame for which to calculate the percentage of missing days.

    Returns
    -------
    float
        The percentage of missing days in the time series DataFrame.
    """

    # Convert the 'data' column to a datetime object
    df_days = pd.DataFrame()
    df_days["data"] = pd.to_datetime(df["data"])
    # Calculate the total number of days in the dataset
    total_days = (df_days["data"].max() - df_days["data"].min()).days + 1
    # Calculate the number of unique days in the dataset
    unique_days = df_days["data"].nunique()
    # Calculate the number of missing days
    missing_days = total_days - unique_days
    # Calculate the percentage of missing days
    percentage_missing_days = (missing_days / total_days) * 100
    return percentage_missing_days


def remove_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using the interquartile range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to remove outliers.
    column : str
        The name of the column in the DataFrame from which to remove outliers.

    Returns
    -------
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    # Calculate the first quartile (Q1)
    Q1 = df[column].quantile(0.25)
    # Calculate the third quartile (Q3)
    Q3 = df[column].quantile(0.75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Calculate the lower bound
    lower_bound = Q1 - 1.5 * IQR
    # Calculate the upper bound
    upper_bound = Q3 + 1.5 * IQR
    # Remove outliers from the DataFrame
    df_iqr = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_iqr


def remove_outliers_zscore(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Remove outliers from a DataFrame using the Z-score method.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame from which to remove outliers.
    column : str
        The name of the column in the DataFrame from which to remove outliers.

    Returns
    -------
    pd.DataFrame
        The DataFrame with outliers removed.
    """
    # Calculate the Z-score for each data point in the column
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    # Remove outliers from the DataFrame
    df_zscore = df[abs(z_scores) < 3]
    return df_zscore


def merge_datasets(
    data: pd.DataFrame | list[pd.DataFrame], on: str | list[str], suffixes=("_x", "_y")
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on a common column.

    Parameters
    ----------
    df1 : list[pd.DataFrame]
        List of DataFrames to merge.
    on : str
        The name of the common column on which to merge the DataFrames.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame.
    """
    from functools import reduce

    df_merged = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=on, suffixes=("", "_")), data
    )
    return df_merged


def skewness(data):
    from scipy.stats import skew

    return skew(data)


def kurtosis(data):
    from scipy.stats import kurtosis

    return kurtosis(data)


def normality_test(data):
    from scipy.stats import shapiro

    return shapiro(data)


def plot_normality(data):
    from scipy import stats
    import matplotlib.pyplot as plt

    stats.probplot(data, dist="norm", plot=plt)
    plt.show()


def data_distribution(data: list[DataFrame], columns: list[str]):
    from scipy import stats
    import pandas as pd
    from matplotlib import pyplot as plt
    from functools import reduce
    import seaborn as sns

    df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on="data", suffixes=("", "_")), data
    )
    df.drop("data", axis=1, inplace=True)
    df.columns = columns

    plt.figure(figsize=(17, 13))
    for i in list(enumerate(df.columns)):
        plt.subplot(2, len(data), i[0] + 1)
        sns.histplot(data=df[i[1]], kde=True)  # Histogram with KDE line

    for i in list(enumerate(df.columns)):
        plt.subplot(2, len(data), i[0] + 1 + len(data))
        stats.probplot(df[i[1]], dist="norm", plot=plt)  # QQ Plot
        plt.title("")

    plt.tight_layout()
    plt.show()


def log_transform(dfs: list[DataFrame], pollutants: list):
    import numpy as np
    from matplotlib import pyplot as plt
    import seaborn as sns

    fig, axs = plt.subplots(len(dfs), 4, figsize=(15, 6))

    for i, data in enumerate(dfs):
        df_log = np.log(data["valore"])
        df_sqrt = np.sqrt(data["valore"])
        df_cbrt = np.cbrt(data["valore"])

        skewness_before = skewness(data["valore"])
        skewness_after_log = skewness(df_log)
        skewness_after_sqrt = skewness(df_sqrt)
        skewness_after_cbrt = skewness(df_cbrt)

        sns.histplot(data["valore"], kde=True, color="red", ax=axs[i][0], legend=False)
        axs[i][0].set(
            xlabel=f"{pollutants[i]}\nSkewness:{round(skewness_before, 2)}\nKurtuosis:{kurtosis(round(data['valore'], 2))}"
        )

        sns.histplot(
            df_log, bins=20, kde=True, ax=axs[i][1], color="orange", legend=False
        )
        axs[i][1].set(
            xlabel=f"{pollutants[i]}\nSkewness:{round(skewness_after_log, 2)}\nKurtuosis:{round(kurtosis(df_log), 2)}"
        )

        sns.histplot(df_sqrt, kde=True, color="green", ax=axs[i][2], legend=False)
        axs[i][2].set(
            xlabel=f"{pollutants[i]}\nSkewness:{round(skewness_after_sqrt, 2)}\nKurtuosis:{round(kurtosis(df_sqrt), 2)}"
        )

        sns.histplot(df_cbrt, kde=True, color="blue", ax=axs[i][3], legend=False)
        axs[i][3].set(
            xlabel=f"{pollutants[i]}\nSkewness:{round(skewness_after_cbrt, 2)}\nKurtuosis:{round(kurtosis(df_cbrt), 2)}"
        )

    axs[0][0].set_title("Original", fontsize=15)
    axs[0][1].set_title("Log", fontsize=15)
    axs[0][2].set_title("Square Root", fontsize=15)
    axs[0][3].set_title("Cube Root", fontsize=15)
    plt.show()


def boxplot_dfs(data: list[DataFrame], columns: list[str]):
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.concat(data, axis=1)
    df.columns = columns

    sns.boxplot(data=df)
    plt.show()


def standardize_data(data: list[DataFrame], columns: list[str]):
    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    df = pd.concat(data, axis=1)
    df.columns = columns

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=columns)

    return df_scaled


def main():
    df = pd.read_csv("data/_processed/Brera/PM10.csv")

    """ df.hist(column="PM10", bins=20)
    plt.show() """

    df1 = standardize_data([df["PM10"]], ["PM10"])
    df1.hist(column="PM10", bins=20)
    plt.show()


if __name__ == "__main__":
    main()
