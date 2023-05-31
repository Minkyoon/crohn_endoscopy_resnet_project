
#데이터 로더 만들기

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
#Preprocess : Removing black area of images


def remove_black_area(image : np.array, tol : int = 20) :
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mask = gray_image>tol
    return image[np.ix_(mask.any(1),mask.any(0))]
#file load
dcm = pydicom.dcmread('/home/minkyoon/crom/20230508_colono_0/3407051/1.2.840.423.140800731320140801115426.2.2_0001_000001_168353251515a4.dcm')
num = dcm.AccessionNumber
img = dcm.pixel_array
print(img.shape)
print(img)
plt.imshow(img, cmap=plt.cm.bone)
if np.all(img[0][0] == np.array([0, 128, 128])):
    color_fixed_image = cv2.cvtColor(img, cv2.COLOR_YUV2RGB)
else:
    color_fixed_image = img
# removed_image = remove_black_area(img)
# plt.imshow(removed_image, cmap=plt.cm.bone)
# removed_image.shape
plt.imshow(color_fixed_image, cmap=plt.cm.bone)
removed_colorfixed_image = remove_black_area(color_fixed_image)
plt.imshow(removed_colorfixed_image, cmap=plt.cm.bone)
# Preprocess 2. Crop the circular endoscopic area and mask
def circular_crop_and_mask(image: np.array):
    height, width, _ = image.shape
    mask = np.zeros((height, width), np.uint8)
    # 이미지 정중앙점 기준으로 원형 크롭 진행
    # 중앙점 찾기 위해서 이미지의 높이, 너비 2로 나눠서 중심점 계산
    cx, cy = width // 2, height // 2
    radius = cy
    cy2 = int(cy *1.18)
    # Create circular mask
    cv2.circle(mask, (width -cy2, cy), radius, 255, -1)
    circular_masked_img = cv2.bitwise_and(image, image, mask=mask)
    # Crop the rectangular bounding box around the circle
    cropped_img = circular_masked_img[cy - radius:cy + radius, width -cy2 - radius:width -cy2 + radius]
    return cropped_img, circular_masked_img
cropped_removed_colorfixed_image, circular_masked_image = circular_crop_and_mask(removed_colorfixed_image)
circular_crop_and_mask
cropped_removed_colorfixed_image
plt.imshow(cropped_removed_colorfixed_image, cmap=plt.cm.bone)
removed_image = remove_black_area(img)
plt.imshow(removed_image, cmap=plt.cm.bone)
removed_image.shape
directory  = ‘/home/jsy/2023_colonoscopy/data/processed/3407051/’
#Preprocess 2. Resize
#####
#Save as npy file
def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print(“Error: Failed to create the directory.“)
createDirectory(directory)
np.save(directory + ‘/1.2.840.423.140800731320140801115426.2.2_0001_000001_168353251515a4.npy’, removed_image)
#npy 파일 불러오기
np.load(‘/home/jsy/2023_colonoscopy/data/processed/3407051/1.2.840.423.140800731320140801115426.2.2_0001_000001_168353251515a4.npy’)
#imagenet.......pretrained model
#Resnet - classification - 내시경 소견 상 호전 / 아님을 맞추는 모델
#Resnet - classification - Hb 7.0 미만 / 이상을 맞추는 모델
#Resnet - regression - Hb 맞추는 모델
#Resnet - regression - Hb, AST, CRP, calprotectin.... 10개 변수를 동시에 맞추는 모델
#data augmentation
#data 지울 것 지우기
#빠트린 것들 다시 차곡차곡
#multiple instance learning


# %%
