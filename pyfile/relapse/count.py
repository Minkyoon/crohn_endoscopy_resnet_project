import pandas as pd

# CSV 파일을 읽습니다.
df = pd.read_csv('/home/minkyoon/crom/pyfile/relapse/relapse.csv')

# 'label' 열에 따라 데이터를 그룹화하고 각 그룹의 개수를 계산합니다.
label_counts = df['label'].value_counts()

print(label_counts)

import numpy as np

image = np.load('/home/minkyoon/2023_crohn/data/relapse2_class_data/0/2203074850b25.npy').astype(np.float32)
import matplotlib.pyplot as plt

# 데이터 로드
image = np.load('/home/minkyoon/2023_crohn/data/relapse2_class_data/0/2203074850b25.npy').astype(np.float32)

# 이미지 플로팅
plt.imshow(image, cmap='gray')  # 흑백 이미지라면 cmap='gray'를 사용하세요.
plt.colorbar()  # 색상 막대 추가 (필요하다면)
plt.show()
image*100