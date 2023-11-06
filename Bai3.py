import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


img = cv.imread('anh2.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Không tìm thấy file"

edges = cv.Canny(img, 100, 200)
ret, thresholded = cv.threshold(edges, 0, 1, cv.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8) 

dilated = cv.dilate(thresholded, kernel, iterations=1)
cleaned_image = cv.erode(dilated, kernel, iterations=1)

plt.subplot(121), plt.imshow(thresholded, cmap='gray')
plt.title('Ảnh gốc'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cleaned_image, cmap='gray')
plt.title('Ảnh đã xử lý'), plt.xticks([]), plt.yticks([])
plt.show()
