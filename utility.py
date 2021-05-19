import os
import sys
import time
import random
import string
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data

#if torch.cuda.is_available():
    #device = torch.device('cuda')
#else:
    #device = torch.device('cpu')
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class convert(object):
    def __init__(self, character, type_to_convert = "ctc"):
        
        self.batch_size = 100
        self.type_to_convert = type_to_convert
        self.image_text = []
        self.character_lists = []
        
        character = list(character)
        self.dictionary = {}
        
        if self.type_to_convert != "att":
            for i, ii in enumerate(character):
                self.dictionary[ii] = i + 1
            self.character = ['[CTC]'] + character
            
        else:
            self.character = ['[GO]', '[s]'] + character
            for i, ii in enumerate(self.character):
                self.dictionary[ii] = i
            
            
    def encode_image(self, image_text):
        
        if self.type_to_convert == "ctc":
            # Count the number of letters in the text
            length_of_text = [len(i) for i in image_text]
        
            # Calculating the CTC loss calculation (The Connectionist Temporal Classification loss)
            batch_text = torch.LongTensor(len(image_text), self.batch_size).fill_(0)
            
            for i, ii in enumerate(image_text):
                image_text = list(ii)
                # Store letters that are the same
                image_text = [self.dictionary[i] for i in image_text]
                batch_text[i][:len(text)] = torch.LongTensor(image_text)
                
            return batch_text.to(device), torch.IntTensor(length_of_text).to(device)
        
        elif self.type_to_convert == "baidu_warp_ctc":
            # Count the number of letters in the text
            length_of_text = [len(i) for i in image_text]
            image_text = ''.join(length_of_text)
            image_text = [self.dictionary[i] for i in image_text]
            
            return torch.IntTensor(image_text), torch.IntTensor(length_of_text)
        
        elif self.type_to_convert == "att":
            # Count the number of letters in the text
            length_of_text = [len(i) for i in image_text]
            # Considering multi-gpu setting
            batch_max_length += 1
            batch_text = torch.LongTensor(len(image_text), self.batch_size).fill_(0)
            
            for i, ii in enumerate(image_text):
                image_text = list(ii)
                image_text.append('[s]')
                image_text = [self.dictionary[i] for i in image_text]
                batch_text[i][1:1 + len(image_text)] = torch.LongTensor(length_of_text)
                
            return batch_text.to(device), torch.IntTensor(length_of_text).to(device)
                
           
    def decode_image(self, image_text_index, length_of_text):
        
        if self.type_to_convert == "standard":
            for i, ii in enumerate(length_of_text):
                # Determine the word it is letter by letter
                word = image_text_index[i, :]
                
                for j in range(ii):
                    # Remove repeating characters and not emphy
                    if word[j] != 0 and (not (j > 0 and word[j - 1] == word[j])):
                        self.character_lists.append(self.character[word[j]])
                    
                words_determine = ''.join(self.character_lists)
                self.image_text.append(words_determine)
                
            return self.image_text
        
        elif self.type_to_convert == "baidu_warp_ctc":
            
            count_index = 0
            for i in length_of_text:
                word = image_text_index[count_index:count_index + 1]
                
                for j in range(i):
                    if word[j] != 0 and (not (j > 0 and word[j - 1] == word[j])):
                        self.character_lists.append(self.character[word[j]])
                        
                    words_determine = ''.join(self.character_lists)
                    self.image_text.append(words_determine)
                    count_index += i
                    
                return self.image_text
            
        elif self.type_to_convert == "att":
            for i, ii in enumerate(length_of_text):
                text = ''.join([self.character[i] for j in image_text_index[i,:]])
                self.image_text.append(text)
                
            return self.image_text
        
        
