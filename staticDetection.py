import cv2
from matplotlib import pyplot as plt

from finalProject.utils.images.imagesUtils import Image
from finalProject.utils.keyPoints.AlgoritamKeyPoints import sift_keypoints_detection
from finalProject.utils.matchers.Matchers import flann_matcher


def gray(path):
    imageclass = Image(path)
    imageclass.read_image(return_img=False)
    return imageclass.gray(return_img=True)


def sobel(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    sobel_intensity = cv2.sqrt(cv2.addWeighted(cv2.pow(sobelx, 2.0),
                                               1.0, cv2.pow(sobely, 2.0), 1.0, 0.0))
    return sobel_intensity


def sobel_keypoints(image):
    sobelImage = sobel(image)
    # norm
    image8bit = cv2.normalize(sobelImage, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    k, d = sift_keypoints_detection(image8bit)
    return k, d, image8bit


def crop(path):
    image = gray(path)
    image = cv2.resize(image, (400, 400))
    x, y, w, h = cv2.selectROI(image, False)
    ROI = image[y:y+h, x:x+w]
    return ROI


path1 = 'dataset/images/DSC_0057.JPG'
path2 = 'dataset/images/DSC_0102.JPG'

image1 = crop(path1)
image2 = crop(path2)

k1, d1, image1 = sobel_keypoints(image1)

k2, d2, image2 = sobel_keypoints(image2)

match = flann_matcher(d1, d2, threshold=0.5)

output = cv2.drawMatchesKnn(image1, k1, image2, k2, match, outImg=None)

plt.figure(figsize=[80, 15])

plt.imshow(output, cmap='gray')
plt.show()
