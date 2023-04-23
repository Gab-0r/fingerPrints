#Dataset: https://www.kaggle.com/datasets/ruizgara/socofing

import numpy as np
import cv2
import fingerprint_enhancer								# Load the library

def create_gaborfilter():
    # This function is designed to produce a set of GaborFilters 
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
     
    filters = []
    num_filters = 16
    ksize = 35  # The local area to evaluate
    sigma = 3.0  # Larger Values produce more edges
    lambd = 10.0
    gamma = 0.5
    psi = 0  # Offset value - lower generates cleaner results
    for theta in np.arange(0, np.pi, np.pi / num_filters):  # Theta is the orientation for edge detection
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F)
        kern /= 1.0 * kern.sum()  # Brightness normalization
        filters.append(kern)
    return filters

def apply_filter(img, filters):
# This general function is designed to apply filters to our image
     
    # First create a numpy array the same size as our input image
    newimage = np.zeros_like(img)
     
    # Starting with a blank image, we loop through the images and apply our Gabor Filter
    # On each iteration, we take the highest value (super impose), until we have the max value across all filters
    # The final image is returned
    depth = -1 # remain depth same as original image
     
    for kern in filters:  # Loop through the kernels in our GaborFilter
        image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
        # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
        np.maximum(newimage, image_filter, newimage)
    return newimage

img = cv2.imread("1.jpg", 1)
cv2.imshow("Huella1", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Huella1 escala de grises", img_gris)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crear elemento estructurante
#kernel = np.ones((1, 1), np.uint8) #Cuadrado
  
# Aplicando erosion 
#img_eroded = cv2.erode(img_gris, kernel) 
#cv2.imshow("Huella1 erosionada", img_eroded)
#cv2.waitKey(0)
#cv2.destroyAllWindows()



img = cv2.imread('1.jpg', 1)						# read input image
out = fingerprint_enhancer.enhance_Fingerprint(img)		# enhance the fingerprint image
cv2.imshow('enhanced_image', out);						# display the result
cv2.waitKey(0)											# hold the display window