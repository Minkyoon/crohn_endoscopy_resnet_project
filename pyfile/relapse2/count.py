import pandas as pd

# CSV 파일을 읽습니다.
df = pd.read_csv('/home/minkyoon/crom/pyfile/relapse2/relapse.csv')

# 'label' 열에 따라 데이터를 그룹화하고 각 그룹의 개수를 계산합니다.
label_counts = df['label'].value_counts()

print(label_counts)