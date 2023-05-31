import pandas as pd

df=pd.read_csv('/home/minkyoon/crom/pyfile/lab_accession_label_20230517.csv')

df=df[['accession_number','anemia']]

grouped_data = df.groupby('anemia')

label_values=[0]


for label_value in label_values:
    group_data = grouped_data.get_group(label_value)
    print(f"Label: {label_value}")
    print(group_data)
    print()

label0=group_data

type(label0)


label_values = [1]  # 확인하려는 라벨 값들

for label_value in label_values:
    group_data = grouped_data.get_group(label_value)
    print(f"Label: {label_value}")
    print(group_data)
    print()
    
label1=group_data

import shutil
import os


# for i in label0['accession_number']:
#     try:
#         path=f'/home/minkyoon/2023_crohn/data/process/{int(i)}'
#         file_list = os.listdir(path)
#         for j in file_list:
#             original = f'/home/minkyoon/2023_crohn/data/process/{int(i)}/{j}'
#             target = f'/home/minkyoon/2023_crohn/data/anemia_class_data/0/{int(i)}b{j}'
#             shutil.copyfile(original, target)

#     except:
#         print(i)
#         continue



for i in label1['accession_number']:
    try:
        path=f'/home/minkyoon/2023_crohn/data/process/{int(i)}'
        file_list = os.listdir(path)
        for j in file_list:
            original = f'/home/minkyoon/2023_crohn/data/process/{int(i)}/{j}'
            target = f'/home/minkyoon/2023_crohn/data/anemia_class_data/1/{int(i)}b{j}'
            shutil.copyfile(original, target)

    except:
        print(i)
        continue
        