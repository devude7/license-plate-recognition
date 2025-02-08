from skimage.segmentation import clear_border
from helpers import filter_ocr, bb_intersection_over_union, diff_accuracy
import numpy as np
import imutils
import cv2
import os
import argparse
import easyocr
import xml.etree.ElementTree as ET

reader = easyocr.Reader(['en'])

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True, help="path to the image folder")
ap.add_argument("-s", "--show", type=bool, help="if you want to see images, and plotted boxes - True")
args = vars(ap.parse_args())

iou_mean = 0
acc = 0
diff_acc = 0

resize_x = 1100
resize_y = 550
images_list = os.listdir(args['images'])

# load the input image from disk
for image in images_list:

    img = cv2.imread(os.path.join('data', image))
    img_count = len(images_list)

    original_height = img.shape[0]
    original_width = img.shape[1]

    img = cv2.resize(img, (resize_x, resize_y))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying morphological operations to locate potential licence plate locations

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #cv2.imshow("Light", light)

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

    thresh_cb = clear_border(thresh)
    #cv2.imshow("thresh-clear", thresh_cb)
    
    thresh_bit = cv2.bitwise_and(thresh_cb, thresh_cb, mask=light)
    #cv2.imshow('thresh-bitwise', thresh_bit)


    # getting potential licence plate contours 
    cnts = cv2.findContours(thresh_bit.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    lpCnt = None
    roi = None

    # loop over the license plate candidate contours with bitwise
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if ar >= 2 and ar <= 5:
            lpCnt = c
            license_plate = gray[y:y + h, x:x + w]
            roi = cv2.threshold(license_plate, 0, 255,
                cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            break
        
    if roi is None or not roi.any():
        print("License Plate not found!")
        
    else:
        roi = clear_border(roi)
        # Resize and erode the image before ocr
        scale_factor = 2.5  
        roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
        roi = cv2.erode(roi, None, iterations=1)

        ocr_lp = reader.readtext(roi, detail = 0)
        if not ocr_lp:
            # if ocr_lp is empty(propably licence was not detected correctly) we'll get potential licence plate contours without bitwise 
            cnts = cv2.findContours(thresh_cb.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

            lpCnt = None
            roi = None

            # loop over the license plate candidate contours without bitwise
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                ar = w / float(h)

                if ar >= 1 and ar <= 6:
                    lpCnt = c
                    license_plate = gray[y:y + h, x:x + w]
                    roi = cv2.threshold(license_plate, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                    break
            if roi is None or not roi.any():
                print("License Plate not found!")
            else:
                roi = clear_border(roi)
                # Resize and erode the image before ocr
                scale_factor = 2.5  
                roi = cv2.resize(roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
                roi = cv2.erode(roi, None, iterations=1)

                ocr_lp = reader.readtext(roi, detail = 0)

        ocr_lp = filter_ocr(ocr_lp[0]) if ocr_lp else 'Number not detected'

        if args['show']:
            cv2.imshow("ROI", roi)
            cv2.imshow("License Plate", license_plate)

        name, ext = os.path.splitext(image)
        tree = ET.parse('annotations.xml')
        root = tree.getroot()
        # calculated licence plate coords
        lp_xmin = x
        lp_ymin = y
        lp_xmax = x + w
        lp_ymax = y + h

        lp_coords = [lp_xmin, lp_ymin, lp_xmax, lp_ymax]

        # we need to rescale org coords as the size of the image has been changed
        scale_x = resize_x / original_width
        scale_y = resize_y / original_height
        
        # annotation coords
        for image in root.findall('.//image'):
            if image.get('name') == name + ext:
                box = image.find('box')
                if box is not None:
                    ann_xmin = int(float(box.get('xtl')) * scale_x)
                    ann_ymin = int(float(box.get('ytl')) * scale_y)
                    ann_xmax = int(float(box.get('xbr')) * scale_x)
                    ann_ymax = int(float(box.get('ybr')) * scale_y)
                    lp_number = box.find('attribute[@name="plate number"]').text            

        ann_coords = [ann_xmin, ann_ymin, ann_xmax, ann_ymax]

        # calculate iou and draw rectangles
        iou = bb_intersection_over_union(lp_coords, ann_coords)
        cv2.rectangle(img, (ann_xmin, ann_ymin), (ann_xmax, ann_ymax), (0, 255, 0), 2)
        cv2.rectangle(img, (lp_xmin, lp_ymin), (lp_xmax, lp_ymax), (0, 0, 255), 2)
        print(f'Intersection over Union (IoU) metric: {iou:.3f}') 
        iou_mean += iou
             
        # accuracy
        print(f"Licence plate: {ocr_lp} == {lp_number}\n")
        if lp_number == ocr_lp:
            acc = acc + 1

        # diff accuracy
        diff = diff_accuracy(ocr_lp, lp_number)
        diff_acc = diff_acc + diff

        if args['show']:
            cv2.imshow("Orginal", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

print(f'Accuracy: {((acc / img_count) * 100):.1f}%')
print(f'Diff Accuracy: {((diff_acc / img_count) * 100):.1f}%')
print(f'Intersection over Union(IoU): {((iou_mean / img_count) * 100):.1f}%')
