import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('./test_image.jpg')
img = np.mean(img, axis=-1).astype(np.uint8)

H, W = img.shape
for h in range(H-2):
    for w in range(W-2):
        window = img[h:h+3, w:w+3]

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img)

plt.show()