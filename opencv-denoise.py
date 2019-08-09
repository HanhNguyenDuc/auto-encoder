import cv2 
import numpy as np 
import os

DIR = 'opencv_data'
for file in os.listdir(DIR):

    img = cv2.imread(os.path.join(DIR, file), 0)
    contrast_img = cv2.medianBlur(img, 5)

    cv2.imshow('blur', contrast_img)
    
    thresh1 = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 75, 13)
    thresh1 = cv2.medianBlur(thresh1, 5)

    cv2.imwrite(os.path.join('output_test_opencv', file), thresh1)