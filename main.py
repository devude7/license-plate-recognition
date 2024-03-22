from skimage.segmentation import clear_border
import pytesseract
import numpy as np
import imutils
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = cv2.imread('data\\images\\Cars1.png')
cv2.imshow('Title', img)

cv2.waitKey(0)

