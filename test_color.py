import matplotlib.pyplot as plt
import cv2


import cv2
image = cv2.imread("dataset/images/color.png")
plt.axis("off")
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()