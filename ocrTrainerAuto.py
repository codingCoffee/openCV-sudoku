import sys
import numpy as np
import cv2


def main():
    ROI_X_Width = 25
    ROI_Y_Height = 35
    ###The size of the training sample
    Training_Image = cv2.imread('trainio.png')
    Training_Output = Training_Image.copy()
    ###Read the traing image
    Modified_Training_Image = cv2.cvtColor(Training_Image,cv2.COLOR_BGR2GRAY)
    Modified_Training_Image = cv2.GaussianBlur(Modified_Training_Image,(5,5),0)
    Modified_Training_Image = cv2.adaptiveThreshold(Modified_Training_Image,255,1,1,11,2)
    Training_Output = cv2.bitwise_not(Training_Output)
    ###Reduce noise and then convert it to binary image
    Contours, Hierarchy = cv2.findContours(Modified_Training_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    ###Find Contours
    Samples =  np.empty((0,ROI_X_Width*ROI_Y_Height))
    ###Creates an empty array of Samples which will be used to store the pixel values
    for Contour in Contours:
        if cv2.contourArea(Contour)>10:
            [Abscissa,Ordinate,X_Width,Y_Height] = cv2.boundingRect(Contour)
            ###Selection criteria for preventing dots(noise)
            if  Y_Height>18:
                ###Another selection criteria (not really required)
                Region_of_Interest = Modified_Training_Image[Ordinate:Ordinate+Y_Height,Abscissa:Abscissa+X_Width]
                Region_of_Interest = cv2.resize(Region_of_Interest,(ROI_X_Width,ROI_Y_Height))
                ###Selects the digits one by one and draws a fitting rectangle around it and waits for a manual keypress
                Sample = Region_of_Interest.reshape((1,ROI_X_Width*ROI_Y_Height))
                Samples = np.append(Samples,Sample,0)
                ###Saves the corresponding pixel values in Samples
    print "Training Complete"
    np.savetxt('Samples.data',Samples)


if __name__ == "__main__":
    main()