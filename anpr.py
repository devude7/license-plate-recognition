from skimage.segmentation import clear_border
from helpers import filter_ocr
import numpy as np
import imutils
import cv2
import os
import argparse
import easyocr

reader = easyocr.Reader(['en'])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, help="path to the image folder")
args = vars(ap.parse_args())

# load the input image from disk
for image in os.listdir(args['images']):

    img = cv2.imread(os.path.join('data', image))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying morphological operations to locate potential licence plate locations

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
        dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = 255 * ((gradX - minVal) / (maxVal - minVal))
    gradX = gradX.astype("uint8")


    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.threshold(gradX, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow('thresh0', thresh)


    thresh = cv2.erode(thresh, None, iterations=2)
    #cv2.imshow('thresh1', thresh)
    thresh = cv2.dilate(thresh, None, iterations=5)
    #cv2.imshow('thresh2', thresh)
    thresh = cv2.erode(thresh, None, iterations=5)
    #cv2.imshow('thresh3', thresh)
    thresh = cv2.dilate(thresh, None, iterations=8)
    #cv2.imshow('thresh4', thresh)

    thresh = clear_border(thresh)
    #cv2.imshow("thresh-clear", thresh)
    thresh = cv2.erode(thresh, None, iterations=4)
    #cv2.imshow('thresh5', thresh)
    thresh = cv2.dilate(thresh, None, iterations=4)
    #cv2.imshow('thresh6', thresh)


    # getting potential licence plate contours 
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    lpCnt = None
    roi = None

    # loop over the license plate candidate contours
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if ar >= 1 and ar <= 6:
            lpCnt = c
            licensePlate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(licensePlate, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            break

    cv2.imshow("License Plate", licensePlate)
    roi = clear_border(roi)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    #cv2.imshow("ROI", roi)

    text = reader.readtext(roi, detail = 0)
    print(f"Licence plate: {filter_ocr(text[0]) if text else 'Not detected'}")

    cv2.waitKey(0)
