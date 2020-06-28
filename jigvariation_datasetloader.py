from .datasetloader import *

from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
import gc

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import gc

from random import seed
from random import randint
from random import random
from random import sample
import numpy as np

from math import sqrt

from google.colab import drive


def pil_loader(path):
    """Open path as file (3 channel RGB)"""
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def pil_to_gray(img, alpha=False):
    """Returns the given (PIL) img input as grayscale image, with or without alpha"""
    
    if alpha:
        return img.convert('LA')
    else:
        return img.convert('L').convert('RGB')
 

def make_decision(probability):
    """Return true or false based on the given probability"""
    return random() < probability

'''
def crop(img, i, n_rows):
    """Returns the cropped i_th image obtained by applying a n_rows*n_rows grid split"""
    
    w, h = img.size
    ws, we  = i%n_rows * w//n_rows, (i+1)%n_rows * w//n_rows
    hs, he  = i//n_rows * h, (i+1)//n_rows * h
    return img.crop(ws, hs, we, he)
'''

def patchize(img, rand_rot_label):
  w, h = img.size
  new_im = Image.new('RGB', (w, h))

  i, j = 0, 0
  
  images_per_row = sqrt(len(rand_rot_label))
  patch_height = int( h//images_per_row )
  patch_width = patch_height

  for num, rotation in enumerate(rand_rot_label):
      if num%images_per_row==0:
          i=0

      # resize my opened image, so it is no bigger than 100,100
      #patch.thumbnail((scaled_img_width,scaled_img_height))
      #Iterate through a 4 by 4 grid with 100 spacing, to place my image
      y_cord = int ( (j//images_per_row)*patch_height )
      patch = img.crop((i, y_cord, i+patch_width, y_cord+patch_height))
      new_patch = patch.rotate(rotation*90)
      #print(rand_rot_label, " R:", rotation)
      new_im.paste(new_patch, (i,y_cord))
      #print(i, y_cord)
      i = (i+patch_width) # +padding
      j += 1
  return new_im

def jig_encode(rand_rot_label):
  """ Encode a list of int in the range 0,3 to a single integer
  >>> jig_encode([3,2,1,0])
  Out: 27
  """

  enc_rot_label = 0
  i = 0
  for l in rand_rot_label:
    enc_rot_label |= l << i*2
    i += 1
  return enc_rot_label

def jig_decode(enc_rot_label, n_rows):
  """ Decode en encoded label to a list of int in the range 0,3 
  >>> jig_decode(27, 2)
  Out: [3, 2, 1, 0]
  """
  rand_rot_label = []
  for i in range(n_rows*n_rows):
    rot_label = (enc_rot_label >> i*2) & 3
    rand_rot_label.append(rot_label)
    i += 1
  return rand_rot_label
     
class ROD_patch(ROD):

    def __init__(self, root, split='train', transform=None, target_transform=None, blacklisted_classes=[], verbose=0,n_samples=0, pre_rotation=False, min_width=0, min_height=0, n_rows=2, n_rand_rotate=2, depth_to_gray_p=0.0):
        super(ROD_patch, self).__init__(root, split=split, transform=transform, target_transform=target_transform, blacklisted_classes=blacklisted_classes, verbose=verbose,n_samples=n_samples, pre_rotation=pre_rotation, min_width=min_width, min_height=min_height)

        # 
        self.n_rows, self.n_rand_rotate = n_rows, n_rand_rotate
        self.depth_to_gray_p = depth_to_gray_p
        
        assert(n_rand_rotate <= n_rows*n_rows)

        patch_label = []
        seed(42)

        patch_list = [x for x in range(n_rows*n_rows)]

        for my_sample in self.data["rgb"]:
            if n_rand_rotate is not None and n_rand_rotate > 0:
                # Get n_rand_rotate random patches to rotate
                patch_to_rotate = sample(patch_list, n_rand_rotate)

            patch_rotation_list = []
            for x in patch_list:
                if x in patch_to_rotate:
                    # Get the random rotation 0->270 that should be applied to the patch
                    patch_rotation_list.append(randint(0, 3))
                else:
                    patch_rotation_list.append(0)
            # Encode 
            enc_rand_rot_label = jig_encode(patch_rotation_list)
            patch_label.append(enc_rand_rot_label)

        self.data["patch_rotations_label"] = patch_label

        # Encode the relative rotation (0, 1, .. ,255) if  n_rows=2 
        le2 = preprocessing.LabelEncoder()
        self.le2 = le2
        self.data['encoded_patch_rotations'] = self.le2.fit_transform(self.data["patch_rotations_label"])

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (rgb_image, depth_image, enc_label, enc_patch_rotations) where target is class_index of the target class.
        '''

        rgb_image, depth_image, label, patch_rotations_label, enc_patch_rotations = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['patch_rotations_label'], self.data.iloc[index]['encoded_patch_rotations'] # Provide a way to access image and label via index
        
        patch_rotations = jig_decode(patch_rotations_label, self.n_rows)
        # Create the patch composed image
        t_depth_image =  patchize(depth_image, patch_rotations)
        
        # Apply gray transf if required
        if self.depth_to_gray_p > 0.0 and self.depth_to_gray_p <= 1.0:
            if make_decision(self.depth_to_gray_p):
              t_depth_image = pil_to_gray(t_depth_image)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            #t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)            
            
        return rgb_image, depth_image, t_depth_image, label, enc_patch_rotations

    def _decode_enc_rotation(self, enc_patch_rotations):
        """enc_patch_rotations given as list, use [x] for single value"""
        return jig_decode(self.le2.inverse_transform(enc_patch_rotations), self.n_rows)
    
    def get_howmany_rot_labels(self, enc_patch_rotations):
        p2_labels = np.zeros(enc_patch_rotations.shape, dtype=np.int32)
        for i, enc_rotations in enumerate(enc_patch_rotations):        
            rotations = self._decode_enc_rotation([enc_rotations])
            for rotation in rotations:
              if rotation > 0:  # Note: rotation is an array (1 elemnt), but it's ok to check this way
                p2_labels[i] += 1 
        return p2_labels
    
class SynROD_patch(SynROD):

    def __init__(self, root, transform=None, target_transform=None, blacklisted_classes=[], verbose=0, n_samples=None, pre_rotation=False, min_width=0, min_height=0, n_rows=2, n_rand_rotate=2, depth_to_gray_p=0.0):
        
        """ Dataloader constructor.
        
        params:
        verbose: set a value >1 to have some useful infos.
        n_sample: Load a given number of sample with a random sapling (without replacement) tecnique. Reduce memory footprint.
        n_rows: Number of rows. The dataloader will create a square n_rows*n_rows grid and rotate :n_rand_rotate patches from 0 to 270 degrees.
        n_rand_rotate: Number of patches to be random rotated.
        depth_to_gray_p: Probability to convert a depth rgb image to grayscale (mantaining rgb channels)
        """
        
        super(SynROD_patch, self).__init__(root, transform=transform, target_transform=target_transform, blacklisted_classes=blacklisted_classes, verbose=verbose,n_samples=n_samples, pre_rotation=pre_rotation, min_width=min_width, min_height=min_height)

        self.n_rows, self.n_rand_rotate = n_rows, n_rand_rotate
        self.depth_to_gray_p = depth_to_gray_p
        
        assert(n_rand_rotate <= n_rows*n_rows)

        patch_label = []
        seed(42)

        patch_list = [x for x in range(n_rows*n_rows)]

        for my_sample in self.data["rgb"]:
            if n_rand_rotate is not None and n_rand_rotate > 0:
                # Get n_rand_rotate random patches to rotate
                patch_to_rotate = sample(patch_list, n_rand_rotate)

            patch_rotation_list = []
            for x in patch_list:
                if x in patch_to_rotate:
                    # Get the random rotation 0->270 that should be applied to the patch
                    patch_rotation_list.append(randint(0, 3))
                else:
                    patch_rotation_list.append(0)
            # Encode 
            enc_rand_rot_label = jig_encode(patch_rotation_list)
            patch_label.append(enc_rand_rot_label)

        self.data["patch_rotations_label"] = patch_label  # This is an integer encoded value 

        # Encode the relative rotation (0, 90, ...)
        le2 = preprocessing.LabelEncoder()
        self.le2 =  preprocessing.LabelEncoder()
        self.data['encoded_patch_rotations'] = self.le2.fit_transform(self.data["patch_rotations_label"])
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample,depth_image, target) where target is class_index of the target class.
        '''

        rgb_image, depth_image, label, patch_rotations_label, enc_patch_rotations = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['patch_rotations_label'], self.data.iloc[index]['encoded_patch_rotations'] # Provide a way to access image and label via index
        patch_rotations = jig_decode(patch_rotations_label, self.n_rows)
        # Create patch composed image (with rotated patches)
        t_depth_image =  patchize(depth_image, patch_rotations)
        # Apply gray transf if required
        if self.depth_to_gray_p > 0.0 and self.depth_to_gray_p <= 1.0:
            if make_decision(self.depth_to_gray_p):
              t_depth_image = pil_to_gray(t_depth_image)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            #t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)          
            
        return rgb_image, depth_image, t_depth_image, label, enc_patch_rotations

    def _decode_enc_rotation(self, enc_patch_rotations):
        """enc_patch_rotations given as list, use [x] for single value"""

        return jig_decode(self.le2.inverse_transform(enc_patch_rotations), self.n_rows)
    
    def get_howmany_rot_labels(self, enc_patch_rotations):
        p2_labels = np.zeros(enc_patch_rotations.shape, dtype=np.int32)
        for i, enc_rotations in enumerate(enc_patch_rotations):        
            rotations = self._decode_enc_rotation([enc_rotations])
            for rotation in rotations:
              if rotation > 0:  # Note: rotation is an array (1 elemnt), but it's ok to check this way
                p2_labels[i] += 1 
        return p2_labels
