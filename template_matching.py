import numpy as np
import cv2
import os
import functools

LABEL_LEN = 9
INPUT_FOLDER = 'opencv_data'
OUTPUT_FOLDER = 'predicted_series_template_matching'
TEMPLATE_FOLDER = 'digit_template'

def template_predicting(filename):
    cell_confidence = np.array(np.zeros(9))
    cell_meaning = list(range(9))
    img = cv2.imread(os.path.join(INPUT_FOLDER, filename), 1)
    cell_width = img.shape[1] // 9
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    contrast_img = cv2.medianBlur(gray, 1)
    
    gray = cv2.adaptiveThreshold(contrast_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv2.THRESH_BINARY,63,13)
    gray = cv2.medianBlur(gray, 5)


    for i in os.listdir(TEMPLATE_FOLDER):
        template = cv2.imread(os.path.join(TEMPLATE_FOLDER, i), 0)
        rate = template.shape[0] / img.shape[0]
        
        temp_shape = template.shape
        template = cv2.resize(template, (int(temp_shape[1] / rate), img.shape[0]), interpolation = cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.0
        loc = np.where(res >= threshold)
        acc = np.where(res >= threshold, res, 0)
        for pt in zip(*loc[::-1]):
            cell = pt[0] // cell_width
            conf = acc[pt[1]][pt[0]]
            if conf >= cell_confidence[cell]:
                cell_confidence[cell] = conf
                cell_meaning[cell] = i[0]
    
    res = functools.reduce(lambda x, y: x + y, cell_meaning)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, filename[:-4] + '_' + res + '.jpg'), img)

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    for f in os.listdir(INPUT_FOLDER):
        template_predicting(f)
