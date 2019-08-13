import cv2 
import pytesseract
import os

fileoutput = open('output_modelencoderdenoise.csv', 'w')
fileoutput.write('image, name \n')
DIR = 'output_test_opencv'
DIR_OUT = 'opencv_output_tesseract'

i = 0
for file in os.listdir(DIR):
    img = cv2.imread(os.path.join(DIR, file))
    config = ('-l eng --oem 1 --psm 6')
    text = pytesseract.image_to_string(img, config = config)
    fileoutput.write(file + ',' + text + '\n')
    print('{} images has loaded'.format(i))
    cv2.imwrite((os.path.join(DIR_OUT, file[:-4]) + '_' + text + '.png'), img)
    i+=1