import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./anh1.png', cv2.IMREAD_GRAYSCALE)


hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

hist_original /= hist_original.sum()


h1 = np.array([0.0] * 128 + [0.5] * 128)  
h2 = np.array([0.0] * 64 + [0.25] * 128 + [0.75] * 64)  
h3 = np.array([0.5] * 128 + [1.0] * 128) 

h1 = np.cumsum(h1)
h2 = np.cumsum(h2)
h3 = np.cumsum(h3)


def histogram_specification(image, h_target):

    lut = np.interp(np.arange(256), np.arange(256), h_target)

    matched_image = lut[image]

    return matched_image

output_image_h1 = histogram_specification(image, h1)
output_image_h2 = histogram_specification(image, h2)
output_image_h3 = histogram_specification(image, h3)

plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(2, 2, 2)
plt.hist(output_image_h1.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram của h1')

plt.subplot(2, 2, 3)
plt.hist(output_image_h2.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram của h2')

plt.subplot(2, 2, 4)
plt.hist(output_image_h3.ravel(), bins=256, range=(0, 256), density=True)
plt.title('Histogram của h3')

plt.tight_layout()
plt.show()
