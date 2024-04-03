# **License Plate Recognition**
Automatic license/number plate recognition using `python` and `opencv`. The approach involves a simple combination of morphological operations for locating plates and `EasyOCR` for character recognition. Evaluation is based on the **intersection over union** metric.


![anpr_car](https://github.com/devude7/license-plate-recognition/assets/112627008/5b1d609c-a78a-4ed0-8020-77dc4bf5b2d2)

# **Data**

The data comes from - [kaggle/car-plate-detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)

I manually selected 20 photos that seem quite good and realistic. The database itself is not completely fitting in my opinion, there are duplicates in it and some photos are completely unsuitable. However, if you look carefully, you can find decent images.

**Now it's important to keep in mind that most images vary a lot**! These differences are pivotal in such a straightforward approach, as selecting and fine-tuning morphological operations to suit all cases proves to be challenging.

# **Approach**
1. **Preprocessing**

First we need to pass arguments -i for path to images directory and optionally -a for annotations directory (I assume that annotations have same name as image they describe) if we want the IOU metric.

Convert images to grayscale, apply blackhat to highlight dark objects on a white background (19x5 kernel as licence plate are wider than they are tall), and use Sobel to  emphasize the edges. Additionally, I create mask that we will apply later on. 

2. **Plate Localization**

Next, blur images, perform closing operations, and thresholding. Erosion and dilation operations are then applied, followed by border cleaning and bitwise operations with previously created mask. All these operations should help us finding areas where licence plate could be located.

3. **Candidate Selection**

Now, take contours of each object on the image after all operations and loop over them in search for suitable candidate by width/height ratio for our licence plate. If we find one, we take the region of interest, next we apply threshold on it for OCR. Worth noting that in case where we won't find suitable candidate I try to find it without bitwise (the image after border clear) operation as for some examples it works better without it.


4. **OCR**

Apply OCR on the region of interest, filtering characters and displaying the results. Before OCR, additional preprocessing steps include **border clearing, resizing, and erosion**.

5. **Intersection Over Union Metric**

If the -a argument is passed, the program will calculate the intersection over union metric based on coordinates provided in XML files in the database and our locally calculated coordinates derived from the license plate candidate contours. Additionally, rectangles are plotted onto the picture to provide a clearer visualization of both sets of coordinates.

# **Results**
**18 out of 20** plates correctly identified. Among these, 16 had only the license plate itself(high IOU score), while 2 contained additional areas(low IOU score).


**8 out of 20** OCR results were completely accurate, some having only one character error and others showing multiple errors. I don't know why the errors occur, as some OCR results aren't correct even though you can cleary see all characters on image passed to OCR.

Achieved an average IOU metric of **0.70**, indicating good results.

To sum it up, the results are solid. We have to keep in mind that real-world scenarios typically involve similar images in the database, facilitating adjustments to the process. This straightforward approach can excel in a stable environment. I assume that using more advanced techniques like DL could yield even better results.