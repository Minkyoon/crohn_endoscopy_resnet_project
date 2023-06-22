import pandas as pd

# 파일 읽기
df = pd.read_csv('output.csv')

# slide_id 컬럼 추가하기
df['slide_id'] = df['pt_filepath'].apply(lambda x: x.split('/')[-1].split('.')[0])

# case_id 컬럼 추가하기
df['case_id'] = ['patient_'+str(i) for i in range(df.shape[0])]

# 새로운 순서로 컬럼 재정렬하기
df = df[['case_id', 'slide_id', 'label']]


# 새로운 파일로 저장하기
df.to_csv('new_output.csv', index=False)