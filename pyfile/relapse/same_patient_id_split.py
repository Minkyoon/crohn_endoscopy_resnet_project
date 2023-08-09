import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('/home/minkyoon/crom/pyfile/relapse/relapse.csv')


data['accession_number'] = data['filepath'].apply(lambda x: x.split('/')[-1].split('b')[0])


unique_accession_numbers = data['accession_number'].unique()

rest_acc_nums, test_acc_nums = train_test_split(unique_accession_numbers, test_size=0.8, random_state=42)
valid_acc_nums, train_acc_nums = train_test_split(rest_acc_nums, test_size=0.5, random_state=42)

train_data = data[data['accession_number'].isin(train_acc_nums)]
valid_data = data[data['accession_number'].isin(valid_acc_nums)]
test_data = data[data['accession_number'].isin(test_acc_nums)]




train_data.to_csv('train1.csv', index=False)
valid_data.to_csv('valid1.csv', index=False)
test_data.to_csv('test1.csv', index=False)

test_data['accession_number'].unique()