import pandas as pd
from sklearn.model_selection import train_test_split

# CSV 파일 불러오기
data = pd.read_csv('/home/minkyoon/2023_crohn/data/anemia_class_data/anemia.csv')

# 데이터를 train, validation, test로 8:1:1 비율로 분할
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# 분할한 데이터를 다시 CSV 파일로 저장
train_data.to_csv('train.csv', index=False)
valid_data.to_csv('valid.csv', index=False)
test_data.to_csv('test.csv', index=False)