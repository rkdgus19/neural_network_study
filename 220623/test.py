import numpy as np
import matplotlib.pyplot as plt

img = np.array([[0, 0, 0, 0, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255, 0],
                [0, 255, 255, 255 ,0],
                [0, 0, 0, 0, 0]
                ])
H, W = 4, 4

filter_size = 3 # 이미지 이동 거리
img_filtered_x = np.zeros(shape=(H-2, W-2))
img_filtered_y = np.zeros(shape=(H-2, W-2))
img_filtered = np.zeros(shape=(H-2, W-2))

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

for h in range(H-2):
    for w in range(W-2):
        window = img[h:h+filter_size, w:w+filter_size]
        z1 = np.abs(np.sum(window*sobel_x))
        img_filtered_x[h, w] = z1
        print(z1)
        z2 = np.abs(np.sum(window * sobel_y))
        img_filtered_y[h, w] = z2
        z = z1**2 + z2**2
        img_filtered[h, w] = z


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 1].imshow(img_filtered_x, cmap='gray')
axes[1, 0].imshow(img_filtered_y, cmap='gray')
axes[1, 1].imshow(img_filtered, cmap='gray')

plt.show()