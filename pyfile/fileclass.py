

# %%

import pandas as pd

df=pd.read_csv('/home/minkyoon/crom/pyfile/relapse_label_20230526.csv')

df=df.iloc[:,[6,7]]
grouped_data = df.groupby('mucosal_healing')

label_values = [0.0]  # 확인하려는 라벨 값들

for label_value in label_values:
    group_data = grouped_data.get_group(label_value)
    print(f"Label: {label_value}")
    print(group_data)
    print()


label0=group_data

label_values = [1.0]  # 확인하려는 라벨 값들

for label_value in label_values:
    group_data = grouped_data.get_group(label_value)
    print(f"Label: {label_value}")
    print(group_data)
    print()


label1=group_data


## 라벨 0부터 옮기자!
import shutil
import os


for i in label0['accession_number']:
    try:
        path=f'/home/minkyoon/2023_crohn/data/process/{int(i)}'
        file_list = os.listdir(path)
        for j in file_list:
            original = f'/home/minkyoon/2023_crohn/data/process/{int(i)}/{j}'
            target = f'/home/minkyoon/2023_crohn/data/class_data/0/{int(i)}b{j}'
            shutil.copyfile(original, target)

    except:
        continue



for i in label1['accession_number']:
    try:
        path=f'/home/minkyoon/2023_crohn/data/process/{int(i)}'
        file_list = os.listdir(path)
        for j in file_list:
            original = f'/home/minkyoon/2023_crohn/data/process/{int(i)}/{j}'
            target = f'/home/minkyoon/2023_crohn/data/class_data/1/{int(i)}b{j}'
            shutil.copyfile(original, target)

    except:
        continue







# %%
