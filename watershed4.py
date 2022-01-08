import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from queue import PriorityQueue  # 优先级队列


def generate_mask(src):
    '''
    from opencv tutorial
    https://docs.opencv.org/master/d2/dbd/tutorial_distance_transform.html
    '''
    src[np.all(src == 255, axis=2)] = 0

    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated
    imgLaplacian = cv.filter2D(src, cv.CV_32F, kernel)
    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    bw = cv.cvtColor(imgResult, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)

    # Dilate a bit the dist image
    kernel1 = np.ones((10, 10), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)

    dist_8u = dist.astype('uint8')

    # Find total markers
    contours = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    contours = contours[1]
    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)
    c = (int)(np.max(markers) + 1)
    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (c, c, c), -1)
    return markers


class Watershed:
    def __init__(self, image, markers):
        self.IN_QUEUE = -2
        self.WSHED = -1  # boundary
        self.q = PriorityQueue()
        self.image = image.copy()
        self.markers = markers.copy()

    def pixel_diff(self, pix1, pix2):
        b = int(self.image[pix1[0]][pix1[1]][0]) - int(self.image[pix2[0]][pix2[1]][0])
        g = int(self.image[pix1[0]][pix1[1]][0]) - int(self.image[pix2[0]][pix2[1]][0])
        r = int(self.image[pix1[0]][pix1[1]][0]) - int(self.image[pix2[0]][pix2[1]][0])
        arr = [np.abs(b), np.abs(b), np.abs(r)]
        return np.max(arr)

    def ws_push(self, diff, pix):
        self.q.put((diff, pix))
        self.markers[pix[0]][pix[1]] = self.IN_QUEUE

    def ws_pop(self):
        if self.q.qsize() > 0:
            pix = self.q.get()[1]
        else:
            pix = [-1, -1]
        return pix

    def pixels_to_push(self, pix):
        i = pix[0]
        j = pix[1]
        if self.markers[i][j] == self.WSHED:
            return

        if self.markers[i - 1][j] == 0:
            diff = self.pixel_diff([i, j], [i - 1, j])
            self.ws_push(diff, [i - 1, j])

        if self.markers[i + 1][j] == 0:
            diff = self.pixel_diff([i, j], [i + 1, j])
            self.ws_push(diff, [i + 1, j])

        if self.markers[i][j - 1] == 0:
            diff = self.pixel_diff([i, j], [i, j - 1])
            self.ws_push(diff, [i, j - 1])

        if self.markers[i][j + 1] == 0:
            diff = self.pixel_diff([i, j], [i, j + 1])
            self.ws_push(diff, [i, j + 1])

    def label_pix(self, coord):
        label = 0
        i = coord[0]
        j = coord[1]
        if self.markers[i - 1][j] > 0:
            if label == 0:
                label = self.markers[i - 1][j]
            elif label != self.markers[i - 1][j]:
                label = self.WSHED

        if self.markers[i + 1][j] > 0:
            if label == 0:
                label = self.markers[i + 1][j]
            elif label != self.markers[i + 1][j]:
                label = self.WSHED

        if self.markers[i][j - 1] > 0:
            if label == 0:
                label = self.markers[i][j - 1]
            elif label != self.markers[i][j - 1]:
                label = self.WSHED

        if self.markers[i][j + 1] > 0:
            if label == 0:
                label = self.markers[i][j + 1]
            elif label != self.markers[i][j + 1]:
                label = self.WSHED

        self.markers[i][j] = label

    def do_water_shed(self):
        # opencv中视边缘为边界boundary pixels
        self.markers[0, :] = self.WSHED
        self.markers[self.markers.shape[0] - 1, :] = self.WSHED
        self.markers[:, 0] = self.WSHED
        self.markers[:, self.markers.shape[1] - 1] = self.WSHED
        '''第一步将markers的初始点放进优先队列'''
        for i in range(1, self.markers.shape[0] - 1):
            for j in range(1, self.markers.shape[1] - 1):
                if self.markers[i][j] == 0 and \
                        (self.markers[i - 1][j] > 0 or self.markers[i + 1][j] > 0 or \
                         self.markers[i][j - 1] > 0 or self.markers[i][j + 1] > 0):
                    '''找与marker最小的梯度'''
                    diff = 255
                    if self.markers[i - 1][j] > 0:
                        diff = min(self.pixel_diff([i, j], [i - 1, j]), diff)
                    if self.markers[i + 1][j] > 0:
                        diff = min(self.pixel_diff([i, j], [i + 1, j]), diff)
                    if self.markers[i][j - 1] > 0:
                        diff = min(self.pixel_diff([i, j], [i, j - 1]), diff)
                    if self.markers[i][j + 1] > 0:
                        diff = min(self.pixel_diff([i, j], [i, j + 1]), diff)
                    self.ws_push(diff, [i, j])
        '''白色为在队列中的点'''
        plt.figure()
        plt.imshow((self.markers == self.IN_QUEUE), "gray")  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        cnt = 0
        '''第二步队列出一个进一次'''
        while self.q.qsize() > 0 and cnt < (self.markers.shape[0] * self.markers.shape[1]):
            cnt = cnt + 1
            pix = self.ws_pop()
            self.label_pix(pix)
            self.pixels_to_push(pix)


def main():
    input = 'D:\imgR\imgR2\SINet-V2\Datasets\TestDataset\Locust6-oringin\Image\\7.jpg'
    src = cv.imread(input)
    # 以opencv官方例子为例，先得到mask
    markers = generate_mask(src)

    # plt.figure()
    # plt.imshow(markers, "gray")  # 显示图片
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()

    ws = Watershed(src, markers)
    ws.do_water_shed()

    plt.figure()
    plt.imshow(ws.markers, "gray")  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    f = plt.gcf()  # 获取当前图像
    f.savefig(r'D:\imgR\imgR2\testdata\{7}.png',transparent=True)
    f.clear()  # 释放内存
    plt.show()



if __name__ == "__main__":
    main()