import numpy as np
from scipy.interpolate import griddata
import cv2
from collections import deque


def imgPreProcess(Sudoku_Image):
    Modified_Sudoku_Image = cv2.cvtColor(Sudoku_Image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Modified_Sudoku_Image.png", Modified_Sudoku_Image)
    ###We convert the image to B&W format to easily be able to extract info
    Modified_Sudoku_Image = cv2.GaussianBlur(Modified_Sudoku_Image,(5,5),0)
    #cv2.imshow("Modified_Sudoku_Image.png", Modified_Sudoku_Image)
    ###Gaussian Blur is applied to remove any noise from the image
    ###https://www.youtube.com/watch?v=C_zFhWdM4ic
    ###cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
    ###(src) ~input image
    ###(ksize) ~kernal size
    ###(sigmaX, sigmaY) ~.indicate the standard deviation in the x and y directions,
    ## .making both of them 0 means the gaussian kernal is automatically calculated
    Modified_Sudoku_Image = cv2.adaptiveThreshold(Modified_Sudoku_Image,255,1,1,19,5)
    #cv2.imshow("Modified_Sudoku_Image.png", Modified_Sudoku_Image)
    ###Adaptive Thresholding is done to adjust for different lighting conditions
    ###cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
    ###(src) ~input image
    ###(maxValue) ~value assigned to pixels for which the contdition is satisfied
    ###(adaptiveMehod) ~ADAPTIVE_THRESH_MEAN_C == 0; ADAPTIVE_THRESH_GAUSSIAN_C == 1
    ###(thresholdType) ~THRESH_BINARY == 0; THRESH_BINARY_INV == 1
    ###(blockSize) ~.something like the kernal size
    ###(C) ~Constant subtracted from the mean or weighted mean ~.to clear the noise
    #cv2.imshow('Modified_Sudoku_Image', Modified_Sudoku_Image)
    return Modified_Sudoku_Image


def findSquare(Modified_Sudoku_Image):
    Temp_Image = Modified_Sudoku_Image.copy()
    #cv2.imshow('Temp_Image', Temp_Image)
    ###Create a copy of the Modified_Sudoku_Image to implement cv2.findContours() on,
    ## since after applying this method the image gets distorted for some reason.
    ###We are using the .copy() method to create the image since using something like
    ## img1 = img2, simply creates an object pointing to the original one. So altering
    ## either of the images also alters the other image and hence using it makes no sense
    Contours, Hierarchy = cv2.findContours(Temp_Image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #OGimg = cv2.cvtColor(Temp_Image,cv2.COLOR_GRAY2RGB)
    #cv2.drawContours(OGimg,Contours,-1,(0,255,0),1)
    #cv2.imshow("Modified_Sudoku_Image.png", OGimg)
    ###Find the contours in the image
    ###cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])
    ###(image) ~input binary image
    ###Refer the link below for more info
    ##http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html#findcontours
    Required_Square_Contour = None
    Required_Contour_Area = 0
    for Contour in Contours:
        Contour_Area = cv2.contourArea(Contour)
        ###Calculates the area enclosed by the vector of 2D points denoted by 
        ## the variable Contour
        if Contour_Area > 500:
            if Contour_Area > Required_Contour_Area:
                Required_Contour_Area = Contour_Area
                Required_Square_Contour = Contour
    ###Code for finding out the largest contour (on the basis of area)
    Perimeter_of_Contour = cv2.arcLength(Required_Square_Contour, True)
    ###Calculates a contour perimeter or a curve length
    ###cv2.arcLength(curve, closed)
    ###(curve) ~Input vector of 2D points
    ###(closed) ~Flag indicating whether the curve is closed or not
    Temp_Square_Countour = cv2.approxPolyDP(Required_Square_Contour, 0.05*Perimeter_of_Contour, True)
    ###Approximates a polygonal curve(s) with the specified precision
    ###cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])
    Temp_Square_Countour = Temp_Square_Countour.tolist()
    Approx_Square_Countour = []
    for Temp_Var_1 in Temp_Square_Countour:
        for Temp_Var_2 in Temp_Var_1:
            Temp_Var_2[0], Temp_Var_2[1] = Temp_Var_2[1], Temp_Var_2[0]
            Approx_Square_Countour.append(Temp_Var_2)
    ###Temp_Square_Countour has the coordinates inside a list within a list, 
    ## hence to extract it we're doing this. Also we're changing (row, column) i.e.
    ## (y, x) to (column, row) i.e. (x, y)
    ###This was done because the griddata function from the scipy library 
    ## takes in values as (column, row) i.e. (x,y) instead of (row, column) i.e (y,x)
    Approx_Square_Countour = deque(Approx_Square_Countour)
    ###Applying deque function on anything converts it into a queue and we can use
    ## functions like popleft() etc on it, as if it were a queue 
    Min_Sum = 9999999
    ###Initialized to a fairly large number as we want minimum
    Counter = -1
    ###Used as counter to keep tract of the iteration number so that the
    ## location of top-left coordinate can be stored in the variable Loc
    Loc = 0
    for i in Approx_Square_Countour:
        Counter+=1
        if Min_Sum > sum(i):
            Min_Sum = sum(i)
            Loc = Counter
    if Loc != 0:
        for i in range(0,Loc):
             Approx_Square_Countour.append(Approx_Square_Countour[0])
             Approx_Square_Countour.popleft()
    ###If the sum of the x and y coordinates is minimum it would automatically
    ## mean that the coordinate refers to the top-left point of the square.
    ###We know the coordinates of the square are stored in a cyclic fashion,
    ## hence if we know the location of the top-left coordinate then we can
    ## re-arrage it by appending the 1st coordinate and then poping it.
    ## Example: (4,1,2,3)
    ## Now appending 1st loc we get (4,1,2,3,4)
    ## Now popping 1st loc we get (1,2,3,4) which is the required result
    ## That is what this code does to rearrange the coordinates
    Approx_Square_Countour[1], Approx_Square_Countour[3] = Approx_Square_Countour[3], Approx_Square_Countour[1]
    ###Flipping the location of 1st and 3rd coordinates makes the coordinate 
    ## pointer go counter-clockwise. We do this because opencv stores the 
    ## coordinate values in a clockwise fashion, however griddata function from 
    ## scipy library requires it to be in a counter-clockwise fashion
    #cv2.drawContours(Modified_Sudoku_Image,[Approx_Square_Countour],0,255,10)
    Mask = np.zeros((Modified_Sudoku_Image.shape),np.uint8)
    ###Creates a black image of the same size as the input image
    cv2.drawContours(Mask,[Required_Square_Contour],0,255,-1)
    cv2.drawContours(Mask,[Required_Square_Contour],0,0,2)
    ###Overwrites the black image with the area of the sudoku in white
    Modified_Sudoku_Image = cv2.bitwise_and(Modified_Sudoku_Image,Mask)
    ###Compares the Modified_Sudoku_Image and the Mask and blackens all parts 
    ## of the image other than the sudoku
    #cv2.imshow('Modified_Sudoku_Image', Modified_Sudoku_Image)
    return Modified_Sudoku_Image, Approx_Square_Countour


def stretchSquare(Modified_Sudoku_Image, Square_Contour):
    grid_x, grid_y = np.mgrid[0:449:450j, 0:449:450j]
    ###Creates grid_x such that it is a 2D array having all values equal to their corresponding row value
    ## Creates grid_y such that it is a 2D array having all values equal to their corresponding column value
    destination = np.array([[0,0],[0,449],[449,449],[449,0]])
    ###Denotes the coordinates of the corners of the destination onto which we want to map the sudoku
    source = np.asarray(Square_Contour)
    ###Denotes the coordinates of the corners of the sudoku as present in the source
    grid_z = griddata(destination, source, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:,1] for ar in grid_z]).reshape(450,450)
    map_y = np.append([], [ar[:,0] for ar in grid_z]).reshape(450,450)
    ###Converts the values to stretch/contract the image and accordingly adjust pixel values
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')
    ###Converts the values to the specified type
    Warped_Sudoku_Image = cv2.remap(Modified_Sudoku_Image, map_x_32, map_y_32, cv2.INTER_CUBIC)
    ###Used to remap the sudoku from the original image into a new image of 
    ## size 450x450, this size was chosen because identifying each small block becomes easier
    ## since it'll have a size of 50x50
    Warped_Sudoku_Image = cv2.bitwise_not(Warped_Sudoku_Image)
    ###Inverts the color scheme
    #cv2.imshow("Warped_Sudoku_Image.png", Warped_Sudoku_Image)
    cv2.imwrite("Temp_Storage/Warped_Sudoku_Image.png", Warped_Sudoku_Image)
    return Warped_Sudoku_Image


def flowChart(Sudoku_Image):
    Modified_Sudoku_Image = imgPreProcess(Sudoku_Image)
    ###Removes noise and converts the image into binary form i.e. 0's and 1's
    Modified_Sudoku_Image, Square_Contour = findSquare(Modified_Sudoku_Image)
    ###Locates the sudoku square and removes all the other stuff around it
    Warped_Sudoku_Image = stretchSquare(Modified_Sudoku_Image, Square_Contour)
    ###Resizes the sudoku square to a new image of 450x450 pixels
    return Warped_Sudoku_Image


def main(file_number):
    ### code for checking output for single image ###
    #'''
    #code for checking output for single image
    Sudoku_Image = cv2.imread('trial-images/sudoku '+str(file_number)+'.png')
    #global OGimg
    #OGimg = Sudoku_Image.copy()
    #cv2.imshow('Sudoku_Image', Sudoku_Image)
    Warped_Sudoku_Image = flowChart(Sudoku_Image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    '''
    ### code for checking output for all images ###
    #no_of_test_images = 5
    for file_number in range(1,no_of_test_images+1):
        file_name = 'trial-images/sudoku '+str(file_number)+'.png'
        Sudoku_Image = cv2.imread(file_name)
        Warped_Sudoku_Image = flowChart(Sudoku_Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #'''
    return Warped_Sudoku_Image


if __name__ == "__main__":
    main(3)
