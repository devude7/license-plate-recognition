# **License Plate Recognition**
Automatic license/number plate recognition using `python` and `opencv`. The approach involves a simple combination of morphological operations for locating plates and `EasyOCR` for character recognition. Evaluation is based on the **intersection over union** metric.

# **Data**

The data comes from our dataset, which we created and labeled ourselves. Published and available on kaggle - [kaggle/poland-vehicle-license-plate-dataset](https://www.kaggle.com/datasets/piotrstefaskiue/poland-vehicle-license-plate-dataset)

The dataset contains 195 photos with annotations in an xml file with license plate coordinates. More information about dataset can be found on kaggle

The photos are mostly similar to each other, which aids in conducting an effective machine learning process and also in finding an appropriate approach using traditional computer vision methods. It is worth mentioning that the photos have high resolution and various sizes (if I am not mistaken, there are two different dimensions of the photos in the database), so it is advisable to resize them to reduce computational requirements. However, when resizing, it is important to also adjust the coordinates to the new dimensions.

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
[In progress]