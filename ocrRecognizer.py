import cv2
import numpy as np


### Used to directly apply OCR based on the data pregenerated ###
def main():
    ROI_X_Width = 25
    ROI_Y_Height = 35
    ###The size of the training sample
    ### Loading the trained data ###
    Samples = np.loadtxt('Samples.data',np.float32)
    Responses = np.loadtxt('Responses.data',np.float32)
    Responses = Responses.reshape((Responses.size,1))
    ### Train using the data ###
    model = cv2.KNearest()
    model.train(Samples,Responses)
    ###We use a model variable because there can be multiple models to train on in the same code
    Warped_Sudoku_Image = cv2.imread('Temp_Storage/Warped_Sudoku_Image.png')
    ###Load the image to process
    Output_Image = np.zeros(Warped_Sudoku_Image.shape,np.uint8)
    ###Create an black image of same size
    Modified_Sudoku_Image = cv2.cvtColor(Warped_Sudoku_Image,cv2.COLOR_BGR2GRAY)
    Modified_Sudoku_Image = cv2.adaptiveThreshold(Modified_Sudoku_Image,255,1,1,11,2)
    ###Reduce noise
    Contours,Hierarchy = cv2.findContours(Modified_Sudoku_Image,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    ###Find Contours
    Sudoku_Text = [[0] * 9 for i in range(9)]
    ###Create a 9x9 list of all zeros i.e. a blank sudoku
    ### Recognition part ###
    for Contour in Contours:
        ###Select a contour one by one
        if cv2.contourArea(Contour)>30 and cv2.contourArea(Contour)<1000:
            ###Selection criteria for preventing selection of dots(noise) and sudoku grid squares
            [Abscissa,Ordinate,X_Width,Y_Height] = cv2.boundingRect(Contour)
            if  Y_Height>22 and Y_Height<40:
                ###Selection criteria for preventing selection of sudoku grid
                cv2.rectangle(Warped_Sudoku_Image,(Abscissa,Ordinate),(Abscissa+X_Width,Ordinate+Y_Height),(0,255,0),2)
                ###Selects the digits one by one and draws a fitting rectangle around it
                Region_of_Interest = Modified_Sudoku_Image[Ordinate:Ordinate+Y_Height,Abscissa:Abscissa+X_Width]
                ###Selects the number in question
                Region_of_Interest = cv2.resize(Region_of_Interest,(ROI_X_Width,ROI_Y_Height))
                ###Resizes it to a 10x10 image
                Region_of_Interest = Region_of_Interest.reshape((1,ROI_X_Width*ROI_Y_Height))
                ###Converts the 10x10 image into an array of 100 pixel values
                Region_of_Interest = np.float32(Region_of_Interest)
                ###Converts it into float32 type
                Retval, Result, Neigh_Resp, Dists = model.find_nearest(Region_of_Interest, k = 1)
                ###Apply kNN algorithm to find nearest neighbours
                string = str(int((Result[0][0])))
                ###Converts the result into an integer and then a string to put on the output image
                cv2.putText(Output_Image,string,(Abscissa,Ordinate+Y_Height),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
                ###cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                ###(img) ~Image.
                ###(text) ~Text string to be drawn.
                ###(org) ~Bottom-left corner of the text string in the image.
                ###(font) ~CvFont structure initialized using InitFont().
                ###(fontFace) ~Font type. One of FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX,
                ## FONT_HERSHEY_COMPLEX, FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX,
                ## or FONT_HERSHEY_SCRIPT_COMPLEX, where each of the font ID's can be combined with FONT_ITALIC to get
                ## the slanted letters
                ###(fontScale) ~Font scale factor that is multiplied by the font-specific base size.
                ###(color) ~Text color
                Sudoku_Text[(Ordinate+Y_Height)/50][Abscissa/50] = int(string)
                ### row=(Ordinate+Y_Height)/50 since it is a 450x450 grid, same for col
                ## Hence Sudoku_Text stores the values of the identified difit
                ## in its respective place
    #print np.asarray(Sudoku_Text)
    #cv2.imshow('Warped_Sudoku_Image',Warped_Sudoku_Image)
    cv2.imwrite('Temp_Storage/Sudoku_Image.png',Warped_Sudoku_Image)
    #cv2.imshow('Output_Image',Output_Image)
    cv2.imwrite('Temp_Storage/Output_Image.png',Output_Image)
    #cv2.waitKey(0)
    #print Sudoku_Text
    return Sudoku_Text


if __name__ == "__main__":
    main()