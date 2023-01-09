import cv2
img = cv2.imread('data/seg/16.PNG',cv2.IMREAD_UNCHANGED)
for i in range(256):  # 给全部的像素点按照阈值划分前景和背景
    for j in range(256):
        if img[i, j] == 0:
            img[i, j] = 255.
        else:
            img[i, j] = 0.
img = cv2.imwrite('data/inverse_pic/inverse_16.PNG', img)