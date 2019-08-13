import cv2 
import numpy as np 
import os

DIR = 'opencv_data'
i = 0

def denoise():
    for file in os.listdir(DIR):
        # i+=1
        # if (i % 1000 == 0):
        #     print('{} images has loaded'.format(i))

        img = cv2.imread(os.path.join(DIR, file), 1)
        img = img[:, :, 0]
        contrast_img = cv2.medianBlur(img, 1)
        
        thresh1 = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,63,13)
        

        # kernel = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype = np.uint8).reshape(3, 3)
        # erosion = cv2.erode(thresh1, kernel, iterations = 1)
        thresh1 = cv2.medianBlur(thresh1, 5)
        
        cv2.imshow('img', thresh1)
        cv2.waitKey()
        cv2.imwrite(os.path.join('output_series_threshold', file), thresh1)

if __name__ == '__main__':
    # split_number()
    denoise()