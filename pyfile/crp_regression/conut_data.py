import pandas as pd

df=pd.read_csv('/home/minkyoon/crom/pyfile/crp_regression/new_file.csv')

nan_count = df.isna().sum()
print(nan_count)

df_cleaned = df.dropna()

# NaN 값의 개수 확인
nan_count = df_cleaned.isna().sum()
print(nan_count)