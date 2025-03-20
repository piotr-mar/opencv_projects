import cv2
from matplotlib import pyplot as plt

cb_img = cv2.imread('images/checkerboard_18x18.png', 0)
print(cb_img)
plt.imshow(cb_img, cmap='gray')
plt.show()

print(cb_img[0, 0])
print(cb_img[1, 6])

cb_img_copy = cb_img.copy()
cb_img_copy[2, 2] = 200
cb_img_copy[2, 3] = 200
cb_img_copy[3, 2] = 200
cb_img_copy[3, 3] = 200
plt.imshow(cb_img_copy, cmap='gray')
plt.show()

img_nz_bgr = cv2.imread("images/New_Zealand_Boat.jpg", cv2.IMREAD_COLOR_BGR)
img_nz_rgb = cv2.cvtColor(img_nz_bgr, cv2.COLOR_BGR2RGB)
# Ablternative
# img_nz_rgb = img_nz_bgr[:, :, ::-1]

plt.imshow(img_nz_rgb)
plt.show()

cropped_region = img_nz_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()

resized_cropped_region_2x = cv2.resize(cropped_region, None, fx=2, fy=2)

plt.imshow(resized_cropped_region_2x)
plt.show()

resized_cropped_region_2x_interp = cv2.resize(cropped_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
plt.imshow(resized_cropped_region_2x_interp)
plt.title("Resize Interpolation")
plt.show()

desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)
resized_cropped_region = cv2.resize(cropped_region, dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.title("Resize dim")
plt.show()

resized_cropped_region_2x = resized_cropped_region_2x[:, :, ::-1]
cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

resized_cropped_region_2x_interp = resized_cropped_region_2x_interp[:, :, ::-1]
cv2.imwrite("resized_cropped_region_2x_interp.png", resized_cropped_region_2x_interp)


img_nz_rgb_flipped_horz = cv2.flip(img_nz_rgb, 1)
img_nz_rgb_flipped_vert = cv2.flip(img_nz_rgb, 0)
img_nz_rgb_flipped_both = cv2.flip(img_nz_rgb, -1)

plt.figure(figsize=(18, 5))
plt.subplot(141);plt.imshow(img_nz_rgb_flipped_horz);plt.title("Vertical Flip");
plt.subplot(142);plt.imshow(img_nz_rgb_flipped_vert);plt.title("Horizontal Flip");
plt.subplot(143);plt.imshow(img_nz_rgb_flipped_both);plt.title("Both Flipped");
plt.subplot(144);plt.imshow(img_nz_rgb);plt.title("Original");

plt.show()