import pytesseract
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os


class framing_images(object):
    def __init__(self, target_word):
        
        # Color
        self.color = (255,0,0)

        # Paths
        self.absolute_path = os.path.abspath(os.getcwd()) 
        self.absolute_path = os.listdir(self.absolute_path)

        # Images input
        self.images = [count for count in glob(self.image_path +'*') if 'jpg' in count]
        
        # Image copy
        self.image_copy = self.images

        # Target word in the text
        self.target_word = target_word

        # Get all data from the image
        self.data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT)

        # Get all occurences of the that word
        self.word_occurences = [ i for i, word in enumerate(self.data["text"]) if word.lower() == self.target_word]

    

    def framing(self):
        
        for occ in word_occurences:

            # Extract the width, height, top and left position for word wanted to detect in image
            width = self.data["width"][occ]
            height = self.data["height"][occ]
            left = self.data["left"][occ]
            top = self.data["top"][occ]
            
            # Surrounding points aroundthe world
            point_1 = (left, top)
            point_2 = (left + width, top)
            point_3 = (left + width, top + height)
            point_4 = (left, top + height)
    
            # Draw rectangle around the image if wanted
            self.image_copy = cv2.line(self.image_copy, point_1, point_2, color=self.color , thickness=2)
            self.image_copy = cv2.line(self.image_copy, point_2, point_3, color=self.color , thickness=2)
            self.image_copy = cv2.line(self.image_copy, point_3, point_4, color=self.color , thickness=2)
            self.image_copy = cv2.line(self.image_copy, point_4, point_1, color=self.color , thickness=2)



class recognize_image(object):
    def __init__(self):




class search_image(object):
    def __init__(self):


