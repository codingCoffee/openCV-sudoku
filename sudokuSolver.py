import cv2
import numpy as np
from dlxsudoku.sudoku import Sudoku


### Used to correct the incorrectly identified numbers ###
def sudokuCorrector(Sudoku_Text):
    ### Takes input from user about the corrections and implements them ###
    print 'It seems I messed up somewhere during digit recognition. Can you help me fix the sudoku grid!'
    print 'Enter the Row Number, Column Number and the correct Number of the place I messed up'
    print 'Enter 0 when done'
    while True:
        cv2.imshow('Output_Image',Output_Image)
        cv2.imshow('Sudoku_Image',Sudoku_Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ###Show the Output Image and the identified numbers
        Input =  int(raw_input())
        if Input==0:
            break
        Row = Input/100
        Column = (Input-Row*100)/10
        Number = Input-Row*100-Column*10
        Sudoku_Text[Row][Column] = Number
        cv2.rectangle(Output_Image, (Column*50, Row*50), (Column*50+50, Row*50+50), 0, thickness=-1)
        ###Draws a filled black rectangle on the square which detected the incorrect number
        cv2.putText(Output_Image,str(Number),(Column*50+10, Row*50+40),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,255,0))
        ###Prints the correct number as given by the user there
    Solution = sudokuSolver(Sudoku_Text)
    return Solution


def sudokuSolver(Sudoku_Text):
    ###Converts Sudoku_Text from a list of lists to a string, since the Suudoku function
    ## takes input as a string
    Sudoku_String = ''.join(str(j) for i in Sudoku_Text for j in i )
    Flag = True
    ### To catch an error generated due to incorrect sudoku being passed ###
    try:
        Solution = Sudoku(Sudoku_String)
    except:
        Flag = False
    if Flag:
        ### To catch an error generated due to sudoku having multiple solutions being passed ###
        try:
            Solution.solve()
        except:
            print "This sudoku has multiple solutions or maybe"
            Solution = sudokuCorrector(Sudoku_Text)
    else:
        Solution = sudokuCorrector(Sudoku_Text)
    return Solution


def printImage(Sudoku_String):
    Count = 0
    for Number in Sudoku_String:
        Row = Count/9
        Column = Count%9
        cv2.putText(Sudoku_Image,str(Number),(Column*50+15, Row*50+35),cv2.FONT_HERSHEY_SCRIPT_COMPLEX,1,(0,0,255), thickness=2)
        #cv2.rectangle(Sudoku_Image, (Column*50, Row*50), (Column*50+50, Row*50+50), (0,0,255))
        Count+=1
    cv2.imwrite('Temp_Storage/Solution_Image.png',Sudoku_Image)
    #cv2.imshow('Solution_Image',Sudoku_Image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



def main(Sudoku_Text):
    '''
    Sudoku_Text = [ [0, 0, 0, 6, 0, 4, 7, 0, 0], 
                    [7, 0, 6, 0, 0, 0, 0, 0, 9], 
                    [0, 0, 0, 0, 0, 5, 0, 8, 0], 
                    [0, 7, 0, 0, 2, 0, 0, 9, 3], 
                    [8, 0, 0, 0, 0, 0, 0, 0, 5], 
                    [4, 3, 0, 0, 1, 0, 0, 7, 0], 
                    [0, 5, 0, 2, 0, 0, 0, 0, 0], 
                    [3, 0, 0, 0, 0, 0, 2, 0, 8], 
                    [0, 0, 2, 3, 0, 1, 0, 0, 0]]
    #'''
    global Output_Image
    global Sudoku_Image
    Output_Image = cv2.imread('Temp_Storage/Output_Image.png')
    Sudoku_Image = cv2.imread('Temp_Storage/Sudoku_Image.png')
    ###Loads the images required as global so that all functions can operate on them
    Solution = sudokuSolver(Sudoku_Text)
    printImage(Solution.to_oneliner())
    return Solution


if __name__ == '__main__':
    Sudoku_Text = [[2, 0, 0, 0, 0, 6, 1, 0, 0], [1, 0, 0, 0, 9, 2, 0, 8, 0], [0, 0, 7, 0, 0, 0, 0, 0, 4], [0, 2, 9, 8, 0, 0, 0, 0, 0], [0, 7, 0, 0, 5, 0, 0, 2, 0], [0, 0, 0, 0, 0, 7, 3, 5, 0], [4, 0, 0, 0, 0, 0, 9, 0, 0], [0, 8, 0, 4, 1, 0, 0, 0, 7], [0, 0, 3, 6, 0, 0, 0, 0, 5]]
    main(Sudoku_Text)