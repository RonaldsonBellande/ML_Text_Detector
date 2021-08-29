import pytesseract
import cv2
import sys
import matplotlib.pyplot as plt
from PIL import Image
import os



class framing_images(object):
    def __init__(self, target_word):
        

        # Paths
        self.absolute_path = os.path.abspath(os.getcwd()) 

        # Images input


        # Target word in the text
        self.target_word = target_word

        # Get all data from the image
        self.data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT)

        # Get all occurences of the that word
        self.word_occurences = [ i for i, word in enumerate(self.data["text"]) if word.lower() == self.target_word]





class recognize_image(object):
    def __init__(self):




class search_image(object):
    def __init__(self):


