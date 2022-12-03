import cv2
import os


# dir of data
directory = "img_F"


for filename in os.listdir(directory):
    src = cv2.imread(os.path.join(directory, filename))
    image = cv2.flip(src, 1)
    cv2.imwrite(os.path.join(directory, filename[0:7] + "-F-" + filename[7:]), image)