import numpy as np
def preprocessing(df, q1=0.25, q3=0.75):
    def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit
    def check_outlier(dataframe, col_name):
        low_limit, up_limit = outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

    def replace_with_thresholds(dataframe, variable):
        low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def drop_high_corr_features(dataframe, threshold=0.6):
        corr_matrix = dataframe.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        print(f"Kaldırılan Sütunlar: {to_drop}")
        return dataframe.drop(columns=to_drop)

    for col in df.select_dtypes(include=['int64', 'float64']).columns:  # Yalnızca sayısal sütunları işler
        if check_outlier(df, col):
            print(f"Aykırı değerler '{col}' sütununda bulundu. Eşik değerlerle değiştiriliyor...")
            replace_with_thresholds(df, col)
    drop_high_corr_features(df)
    return df
