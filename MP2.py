import cv2
import imgProcessing
import ocrRecognizer
import sudokuSolver
import os

def main(no_of_test_images):
	for file_number in range(1,no_of_test_images+1):
		imgProcessing.main(file_number)
		Sudoku_Text = ocrRecognizer.main()
		Solution = sudokuSolver.main(Sudoku_Text)
		print Solution
		print "------------------------------------------------------------"
		Solution_Image = cv2.imread('Temp_Storage/Solution_Image.png')
		# cv2.imshow('Solution',Solution_Image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
	os.remove('Temp_Storage/Output_Image.png')
	os.remove('Temp_Storage/Sudoku_Image.png')
	os.remove('Temp_Storage/Warped_Sudoku_Image.png')
	os.remove('Temp_Storage/Solution_Image.png')


if __name__ == "__main__":
	no_of_test_images = 5
	main(no_of_test_images)
