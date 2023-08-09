import pandas as pd
import os

# 원본 CSV 파일을 읽는다.
df = pd.read_csv('/home/minkyoon/crom/pyfile/crp_regression/regression_crp.csv')

# 파일 이름이 저장된 디렉토리를 지정한다.
dir_path = '/home/minkyoon/2023_crohn/data/process'

# dir_path에서 하위 폴더의 목록을 가져온다.
accession_numbers = os.listdir(dir_path)

# 새로운 DataFrame을 생성한다.
new_df = pd.DataFrame()

# 폴더 이름과 원본 DataFrame에서 정보를 검색하고 새로운 DataFrame에 추가한다.
for num in accession_numbers:
    if int(num) in df['accession_number'].values:
        temp_df = df[df['accession_number'] == int(num)]
        # 각 폴더 내의 모든 .npy 파일에 대해 반복
        for file in os.listdir(os.path.join(dir_path, num)):
            if file.endswith('.npy'):
                temp_df_copy = temp_df.copy()  # 기존 정보 복사
                temp_df_copy['file_path'] = os.path.join(dir_path, num, file) # npy 파일의 경로
                new_df = new_df.append(temp_df_copy)

# 새로운 DataFrame을 CSV 파일로 저장한다.
df = new_df.reindex(columns=['file_path', 'label', 'ID', 'accession_number'])
df.to_csv('new_file.csv', index=False)

###accession number 없에고 할지 안할지는 추후선택!!

# 'accession_number' 열을 제거한다.
df = df.drop(columns=['ID'])

# 변경된 DataFrame을 CSV 파일로 저장한다.
df.to_csv('new_file.csv', index=False)