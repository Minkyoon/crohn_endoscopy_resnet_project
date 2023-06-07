import shutil
import glob

source_dir = "/home/minkyoon/crom/pyfile/relapse/mlp"
destination_dir = "/home/minkyoon/first/testclam/feature/unifeature/pt_files"

# '*.pt' 패턴에 맞는 파일의/ 목록을 가져옵니다.

pt_files = glob.glob(source_dir + '/*.pt')

# 각 파일을 목표 디렉토리로 복사합니다.
for file in pt_files:
    shutil.copy(file, destination_dir)