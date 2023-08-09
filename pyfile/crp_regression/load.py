import numpy as np
import matplotlib.pyplot as plt

# 이미지 파일 경로
image_path = '/home/minkyoon/2023_crohn/data/process/1903317291/00012.npy'

# 이미지 로드
image = np.load(image_path)

# 이미지 그리기
plt.imshow(image)
plt.axis('off')  # 축 제거
plt.show()
import pandas as pd
train=pd.read_csv('test.csv')
nan_counts = train.isna().sum()
print(nan_counts)
train.dropna(inplace=True)
nan_counts = train.isna().sum()
print(nan_counts)
train
train.to_csv('train.csv', index=False)
