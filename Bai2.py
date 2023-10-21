import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('anh2.jpg', cv.IMREAD_GRAYSCALE)


assert img is not None, "File could not be read, check with os.path.exists()"

edges = cv.Canny(img, 100, 200)

ret, thresholded = cv.threshold(edges, 0, 1, cv.THRESH_BINARY)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(thresholded, cmap='gray')
plt.title('Hình ảnh cạnh (nhị phân)'), plt.xticks([]), plt.yticks([])
plt.show()
