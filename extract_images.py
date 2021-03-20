#This file is used to convert image pixels to image png format and make a csv file according to the format needed

import numpy as np
import pandas as pd
import cv2
import argparse
from os import mkdir, chdir, getcwd
from os.path import join


# Width and Height of the image as given in the dataset
(width, height) = 48,48


# Convert each row of 'pixels' column into numpy array of images
def readPixels(pixels):
	# Empty list to store the images
	images = []
	
	for pixelRow in pixels:
		# Split the string of pixels separated by spaces
		pixelRow = [int(pixel) for pixel in pixelRow.split(' ')]
		# Convert this list of pixel values into numpy array
		images.append(np.asarray(pixelRow).reshape(width, height))
	
	# Convert the list of images into numpy array and return
	return np.asarray(images)


if __name__ == '__main__':

	# parser = argparse.ArgumentParser()
	# parser.add_argument('path', help='Path to CSV file')
	# arguments = parser.parse_args()
	path = '../dataset/fer2013.csv'
	dataFrame = pd.read_csv(path)
	

	# Extract the 'pixels' column from the dataFrame
	pixels = dataFrame['pixels']
	# Convert each row of 'pixels' column into numpy array of images
	images = readPixels(pixels)
	
	# Store the images in 'images' folder of the current working directory
	# Make the 'images' folder in current working directory
	
	cwd = getcwd()
	mkdir(join(cwd,'images'))
	
	# Change current working directory to the images folder
	chdir(join(cwd,'images'))
	
	# Insert a column into dataFrame to save name of the files
	dataFrame.insert(3, "Name of the file", True)
	
	# Save all the images in the images folder
	counter = 0
	files = []
	for img in images:
		cv2.imwrite('image_' + str(counter)+'.png', img)
		files.append('image_' + str(counter)+'.png')
		counter += 1
		print(counter)
		
	dataFrame["Name of the file"] = files
	# Save new dataset as emotion_dataset.csv
	dataFrame.to_csv(cwd+"/"+"emotion_dataset.csv")
