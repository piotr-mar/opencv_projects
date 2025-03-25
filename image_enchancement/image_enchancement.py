import cv2
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv2.imread('images/New_Zealand_Coast.jpg', cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()

print(img_rgb.shape)

matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

plt.figure(figsize=[18, 5])
plt.subplot(131); plt.imshow(img_rgb_darker), plt.title("Darker")
plt.subplot(132); plt.imshow(img_rgb), plt.title("Original")
plt.subplot(133); plt.imshow(img_rgb_brighter), plt.title("Brighter")
plt.show()

matrix1 = np.ones(img_rgb.shape, dtype="uint8") * 0.5
matrix2 = np.ones(img_rgb.shape, dtype="uint8") * 1.5

img_lower_contras = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_higher_contras = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

plt.figure(figsize=[18, 5])
plt.subplot(131); plt.imshow(img_lower_contras), plt.title("Lower contras")
plt.subplot(132); plt.imshow(img_rgb), plt.title("Original")
plt.subplot(133); plt.imshow(img_higher_contras), plt.title("Higher contras")
plt.show()

img_read = cv2.imread('images/building-windows.jpg', cv2.IMREAD_GRAYSCALE)

retval, img_tresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=[18, 5])
plt.subplot(121); plt.imshow(img_read, cmap="gray"), plt.title("Orginal")
plt.subplot(122); plt.imshow(img_tresh, cmap="gray"), plt.title("Treshold")
plt.show()

img_read = cv2.imread("images/Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)

retval, img_tresh_glob_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
retval, img_tresh_glob_2 = cv2.threshold(img_read, 150, 255, cv2.THRESH_BINARY)
img_tresh_adapt = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=[18, 15])
plt.subplot(221); plt.imshow(img_read, cmap="gray"), plt.title("Orginal")
plt.subplot(222); plt.imshow(img_tresh_glob_1, cmap="gray"), plt.title("Treshold")
plt.subplot(223); plt.imshow(img_tresh_glob_2, cmap="gray"), plt.title("Treshold_1")
plt.subplot(224); plt.imshow(img_tresh_adapt, cmap="gray"), plt.title("Adaptiv")
plt.show()

img_rec = cv2.imread("images/rectangle.jpg", cv2.IMREAD_GRAYSCALE)

img_cir = cv2.imread("images/circle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[20, 5])
plt.subplot(121);plt.imshow(img_rec, cmap="gray")
plt.subplot(122);plt.imshow(img_cir, cmap="gray")
print(img_rec.shape)
plt.show()

result = cv2.bitwise_and(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

result = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

result = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

img_bgr = cv2.imread("images/coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)

print(img_rgb.shape)

logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]
plt.show()

# Read in image of color cheackerboad background
img_background_bgr = cv2.imread("images/checkerboard_color.png")
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

# Set desired width (logo_w) and maintain image aspect ratio
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))

# Resize background image to sae size as logo image
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

plt.imshow(img_background_rgb)
plt.show()
print(img_background_rgb.shape)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(img_mask, cmap="gray")
plt.show()

img_mask_inv = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inv, cmap="gray")
plt.show()

img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)
plt.imshow(img_background)
plt.show()

img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
plt.imshow(img_foreground)
plt.show()

result = cv2.add(img_foreground, img_background)
plt.imshow(result)
plt.show()

arr1 = np.array([200, 250], dtype=np.uint8).reshape(-1, 1)
arr2 = np.array([40, 40], dtype=np.uint8).reshape(-1, 1)
add_numpy = arr1+arr2
add_cv2 = cv2.add(arr1, arr2)
print(add_numpy)
print(add_cv2)