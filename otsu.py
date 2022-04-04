import numpy as np
import cv2
import matplotlib.pyplot as plt


def mean(hist, start, end):
    result = 0
    valor = np.sum(hist[start:end])
    if valor == 0:
        return 0
    for i in range(start, end):
        result += hist[i] * i
    return result / valor


def otsu(hist, t, color_count):
    SIZE = np.sum(hist)
    BGMean = 0
    BGWeight = np.sum(hist[0:t]) / SIZE
    BGMean = mean(hist, 0, t)

    FGMean = 0
    FGWeight = np.sum(hist[t:]) / SIZE
    FGMean = mean(hist, t, color_count)

    return BGWeight * FGWeight * (BGMean - FGMean) ** 2


image = cv2.imread('harewood.jpg', cv2.IMREAD_GRAYSCALE)
IMG_NUM_PIXELS = image.size
IMG_SHAPE = image.shape
COLOR_COUNT = 256

pixels_distribution = np.bincount(image.flatten(), minlength=256)

threshold_range = range(COLOR_COUNT)
criterias = [otsu(pixels_distribution, th, COLOR_COUNT) for th in
             threshold_range]
best_threshold = threshold_range[np.nanargmax(criterias)] - 1

newImage = np.where(image > best_threshold, 255, 0)

(cv2_treshold, cv2_tresholdImage) = cv2.threshold(image, 0, 255,
        cv2.THRESH_OTSU)
print ('cv2 threshold = ', cv2_treshold)
print ('algorithm = ', best_threshold)

fig = plt.figure(figsize=(10, 7))

fig.add_subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('raw')

fig.add_subplot(1, 3, 2)
plt.imshow(newImage, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('otsu')

fig.add_subplot(1, 3, 3)
plt.imshow(cv2_tresholdImage, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.title('cv2')

plt.show()
