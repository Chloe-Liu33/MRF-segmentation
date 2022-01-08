import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import datetime

# from sklearn.cluster import k_means_
from sklearn.cluster import KMeans

# -*- coding:utf8 -*-
import os
path1 = r'D:\imgR\imgR2\SINet-V2\Datasets\TestDataset\Locust6-oringin\Image'
path2 = r'D:\imgR\imgR2\SINet-V2\Datasets\TestDataset\Locust6-kmeans2\Image'
# path1 = r'D:/imgR/imgR2/SINet-V2/Datasets/TestDataset/COD10K1\Image'
# path2 = r'D:/imgR/imgR2/SINet-V2/Datasets/TestDataset/COD10K1_mrfs/Image'
filelist = os.listdir(path1)
total_time=0
num = 0
for item in filelist:
    start_time = datetime.datetime.now()

    # print('item name is ',item)
    if item.endswith('.jpg'):
        name = item.split('.', 3)[0] + '.' + item.split('.', 3)[1]
        name1 = item.split('.',3)[0]
        src = os.path.join(os.path.abspath(path1), item)
        img = np.array(mpimg.imread(src))
        imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgGray = imgGray / 255
        imgCopy = imgGray.copy()
        # 像素值扁平化，一维化
        imgpixel = (imgCopy.flatten()).reshape((imgGray.shape[0] * imgGray.shape[1], 1))
        kind = 2
        # kmeans = k_means_.KMeans(n_clusters=kind)
        kmeans = KMeans(n_clusters=kind)
        label = kmeans.fit(imgpixel)  #对像素矩阵进行kmeans计算
        imgLabel = np.array(label.labels_).reshape(imgGray.shape) #将kmeans聚类后的结果取出来
        end_time = datetime.datetime.now()
        time_consume = (end_time - start_time).microseconds
        total_time += time_consume
        num +=1
        print(time_consume)
        # for i in range (0,imgGray.shape[0]):
        #     for j in range(0,imgGray.shape[1]):
        #         print(str(imgLabel[i][j])+" ",end='')
        #     print("\n")
        dpi = 300  # 图像分辨率
        fig = plt.figure(frameon=False)
        # # fig.set_size_inches(imgLabel.shape[0] / dpi)
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax1)
        plt.imshow(imgLabel, cmap='gray')
        plt.axis('off')

        # 去除图像周围的白边
        height, width, channels = img.shape
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        dst = os.path.join(os.path.abspath(path2), name1 + '.jpg')
        plt.savefig(dst, bbox_inches="tight", pad_inches=0.0)


        # fig.savefig(dst)
average_time = total_time/num/1000000
print(average_time)