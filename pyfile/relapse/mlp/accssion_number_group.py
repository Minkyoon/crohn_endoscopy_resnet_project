import pandas as pd

# Read the csv file
df = pd.read_csv('/home/minkyoon/crom/pyfile/relapse/test.csv')

# Group by 'accession_number'
groups = df.groupby('accession_number')

# Save each group to a separate csv file
for name, group in groups:
    group.to_csv(f"{name}.csv", index=False)