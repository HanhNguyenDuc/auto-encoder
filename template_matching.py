import numpy as np
import cv2
import os
import functools

LABEL_LEN = 9
INPUT_FOLDER = 'opencv_data'
OUTPUT_FOLDER = 'predicted_series_template_matching'
TEMPLATE_FOLDER = 'small_vt_cards_templates'

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def template_predicting(filename):
    cell_confidence = np.array(np.zeros(9))
    cell_meaning = list(range(9))
    img = cv2.imread(os.path.join(INPUT_FOLDER, filename), 1)
    cell_width = img.shape[1] // 9
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img[:, :, 2]


    contrast_img = cv2.medianBlur(gray, 1)
    ret, gray = cv2.threshold(contrast_img, 140, 255, cv2.THRESH_BINARY)
    gray = cv2.medianBlur(gray, 5)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
    # cv2.imshow('img', gray)
    # cv2.waitKey(0)
    labels = ''
    # print(len(contours))
    
    for cnt in contours:
        x, y, w, h = map(int, cv2.boundingRect(cnt))
        if cv2.contourArea(cnt) > 170 and cv2.contourArea(cnt) < 2000:
            # cv2.imwrite('draft/img' + str(i) + '.png', gray[y: y + h, x: x + w])
            # cv2.waitKey()   
            max_acc = -1
            label = ''

            for i in os.listdir(TEMPLATE_FOLDER):
                template = cv2.imread(os.path.join(TEMPLATE_FOLDER, i), 0)
                template = cv2.resize(template, (w, h), interpolation = cv2.INTER_AREA)
                # cv2.waitKey(0)
                res = cv2.matchTemplate(gray[y: y + h, x: x + w], template, cv2.TM_CCOEFF_NORMED)
                if (max_acc < res[0][0]):
                    max_acc = res[0][0]
                    label = i[0]
            labels += label
    # print(labels)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    return labels

if __name__ == '__main__':
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    for f in os.listdir(INPUT_FOLDER):
        print(template_predicting(f))
        # break
