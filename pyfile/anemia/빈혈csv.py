import os
import pandas as pd

# 이미지 데이터가 있는 디렉토리
root_dir = '/home/minkyoon/2023_crohn/data/anemia_class_data'

# 라벨이름으로 된 폴더들
labels = os.listdir(root_dir)

data = []

# 각 라벨 폴더 내의 이미지 파일들을 확인
for label in labels:
    label_dir = os.path.join(root_dir, label)
    
    # 폴더가 아닌 경우 스킵
    if not os.path.isdir(label_dir):
        continue

    # 각 이미지 파일에 대해
    for file in os.listdir(label_dir):
        # 파일 경로 생성
        file_path = os.path.join(label_dir, file)
        
        # 데이터 리스트에 파일 경로와 라벨 추가
        data.append([file_path, label])

# 데이터를 DataFrame으로 변환
df = pd.DataFrame(data, columns=['filepath', 'label'])

# DataFrame을 csv 파일로 저장
df.to_csv('anemia.csv', index=False)