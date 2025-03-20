import cv2 as cv
import matplotlib.pyplot as plt


cb_img = cv.imread("images/checkerboard_18x18.png")
cb_img_fuzz = cv.imread("images/checkerboard_18x18.png", 0)
cb_img_fuzz_2 = cv.imread("images/checkerboard_18x18.png", -1)
print(cb_img)
print(cb_img.shape)
print(cb_img.dtype)

plt.imshow(cb_img, cmap="gray")
plt.show()

plt.imshow(cb_img_fuzz, cmap="gray")
plt.show()

plt.imshow(cb_img_fuzz_2, cmap="gray")
plt.show()

cola = cv.imread("images/coca-cola-logo.png", 1)

plt.imshow(cola, cmap="gray")
plt.show()
print(cola)

cola_reverse = cola[:, :, ::-1]
print(cola_reverse)
plt.imshow(cola_reverse, cmap="gray")
plt.show()

image_zeland = cv.imread("images/New_Zealand_Lake.jpg", 1)
plt.imshow(image_zeland, cmap="gray")
plt.show()
b, g, r = cv.split(image_zeland)

plt.subplot(411), plt.imshow(r, cmap="gray"), plt.title("RED")
plt.subplot(412), plt.imshow(g, cmap="gray"), plt.title("GREEN")
plt.subplot(413), plt.imshow(b, cmap="gray"), plt.title("BLUE")

img_merge = cv.merge((b, g, r))
plt.subplot(414), plt.imshow(img_merge, cmap="gray"), plt.title("Merged")
plt.show()

plt.subplot(141), plt.imshow(r, cmap="gray"), plt.title("RED")
plt.subplot(142), plt.imshow(g, cmap="gray"), plt.title("GREEN")
plt.subplot(143), plt.imshow(b, cmap="gray"), plt.title("BLUE")

img_merge = cv.merge((b, g, r))
plt.subplot(144), plt.imshow(img_merge, cmap="gray"), plt.title("Merged")
plt.show()

image_rgb = cv.cvtColor(img_merge, cv.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()

img_hsv = cv.cvtColor(img_merge, cv.COLOR_BGR2HSV)
h, s, v = cv.split(img_hsv)

plt.figure(figsize=(20, 5))

plt.subplot(141), plt.imshow(h, cmap="gray"), plt.title("Hue")
plt.subplot(142), plt.imshow(s, cmap="gray"), plt.title("Saturation")
plt.subplot(143), plt.imshow(v, cmap="gray"), plt.title("Value")
plt.subplot(144), plt.imshow(image_rgb, cmap="gray"), plt.title("Original")
plt.show()

cv.imwrite("new_new_zeland.png", image_zeland)
