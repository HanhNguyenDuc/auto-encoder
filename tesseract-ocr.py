import cv2 
import pytesseract
import os

fileoutput = open('output.csv', 'w')
fileoutput.write('image, name \n')
DIR = 'output_test_opencv'

for file in os.listdir(DIR):
    img = cv2.imread(os.path.join(DIR, file))
    config = ('-l eng --oem 1 --psm 6')
    text = pytesseract.image_to_string(img, config = config)
    fileoutput.write(file + ',' + text + '\n')