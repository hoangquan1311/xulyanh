import numpy as np
import cv2 as cv

img = cv.imread('anh2.jpg', cv.IMREAD_GRAYSCALE)
assert img is not None, "Không đọc được file"

edges = cv.Canny(img, 100, 200)
ret, thresholded = cv.threshold(edges, 0, 1, cv.THRESH_BINARY)

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresholded, connectivity=8)

object_lengths = []
for label in range(1, num_labels): 
    object_mask = np.uint8(labels == label)
    object_length = cv.countNonZero(object_mask)
    object_lengths.append((label, object_length))

print("Danh sách đối tượng đường và độ dài:")
for obj_label, obj_length in object_lengths:
    print(f"Đối tượng {obj_label}: Độ dài = {obj_length} điểm ảnh")

cv.imshow('Anh goc', img)
cv.imshow('Anh duoc gan nhan', np.uint8(labels * (255 / num_labels)))

cv.waitKey(0)
cv.destroyAllWindows()
