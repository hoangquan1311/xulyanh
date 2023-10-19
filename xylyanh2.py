import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh gốc
image = cv2.imread('download.jpg', cv2.IMREAD_GRAYSCALE)

# Tạo histogram của ảnh gốc
hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])

# Chuẩn hóa histogram gốc
hist_original /= hist_original.sum()

# Định nghĩa các histogram mục tiêu
h1 = np.array([0.0] * 128 + [0.5] * 128)  # histogram cho h1
h2 = np.array([0.0] * 64 + [0.25] * 128 + [0.75] * 64)  # histogram cho h2
h3 = np.array([0.5] * 128 + [1.0] * 128)  # histogram cho h3

# Chuyển các histogram mục tiêu thành kết quả cộng dồn
h1 = np.cumsum(h1)
h2 = np.cumsum(h2)
h3 = np.cumsum(h3)

# Biến đổi ảnh theo histogram mục tiêu
def histogram_specification(image, h_target):
    # Tính toán ánh xạ biểu đồ
    lut = np.interp(np.arange(256), np.arange(256), h_target)

    # Áp dụng ánh xạ biểu đồ để biến đổi ảnh
    matched_image = lut[image]

    return matched_image

# Biến đổi ảnh gốc bằng histogram mục tiêu
output_image_h1 = histogram_specification(image, h1)
output_image_h2 = histogram_specification(image, h2)
output_image_h3 = histogram_specification(image, h3)

# Hiển thị và so sánh kết quả
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.title('Ảnh gốc')

plt.subplot(2, 2, 2)
plt.imshow(output_image_h1, cmap='gray')
plt.title('Ảnh với histogram h1')

plt.subplot(2, 2, 3)
plt.imshow(output_image_h2, cmap='gray')
plt.title('Ảnh với histogram h2')

plt.subplot(2, 2, 4)
plt.imshow(output_image_h3, cmap='gray')
plt.title('Ảnh với histogram h3')

plt.tight_layout()
plt.show()
