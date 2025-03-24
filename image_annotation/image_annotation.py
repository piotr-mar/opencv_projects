import cv2
from matplotlib import pyplot as plt

image = cv2.imread('images/Apollo_11_Launch.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

image_line = image.copy()

cv2.line(image_line, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA)
image_line = cv2.cvtColor(image_line, cv2.COLOR_BGR2RGB)

plt.imshow(image_line)
plt.show()

image_circle = image.copy()
cv2.circle(image_circle, (900, 500), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)
image_circle = cv2.cvtColor(image_circle, cv2.COLOR_BGR2RGB)
plt.imshow(image_circle)
plt.show()

image_rectangle = image.copy()
cv2.rectangle(image_rectangle, (500, 100), (700, 600), (255, 0, 0), thickness=5, lineType=cv2.LINE_8)
plt.imshow(image_rectangle)
plt.show()

image_text = image.copy()
text = "Apollo 11 Saturn V Lunch, July 16, 1969"
font_scale = 2.3
font_face = cv2.FONT_HERSHEY_PLAIN
font_color = (0, 255, 0)
font_thickness = 2

cv2.putText(image_text, text, (200, 700), font_face, font_scale, font_color, font_thickness)
plt.imshow(image_text)
plt.show()

image_text_negative = image.copy()
text = "Apollo 11 Saturn V Lunch, July 16, 1969"
font_scale = -2
font_face = cv2.FONT_HERSHEY_PLAIN
font_color = (0, 255, 0)
font_thickness = 2

cv2.putText(image_text_negative, text, (200, 700), font_face, font_scale, font_color, font_thickness)
plt.imshow(image_text_negative)
plt.show()