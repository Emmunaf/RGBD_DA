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
from random import sample
import numpy as np

from math import sqrt

from google.colab import drive


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

'''
def crop(img, i, n_rows):
    """Returns the cropped i_th image obtained by applying a n_rows*n_rows grid split"""
    
    w, h = img.size
    ws, we  = i%n_rows * w//n_rows, (i+1)%n_rows * w//n_rows
    hs, he  = i//n_rows * h, (i+1)//n_rows * h
    return img.crop(ws, hs, we, he)
'''

def patchize(img, rand_rot_label):
  new_im = Image.new('RGB', (256, 256))

  i, j = 0, 0
  images_per_row = sqrt(len(rand_rot_label))
  patch_height = img.size//images_per_row
  patch_width = patch_height

  for num, rotation in enumerate(rand_rot_label):
      if num%images_per_row==0:
          i=0
      # resize my opened image, so it is no bigger than 100,100
      #patch.thumbnail((scaled_img_width,scaled_img_height))
      #Iterate through a 4 by 4 grid with 100 spacing, to place my image
      y_cord = (j//images_per_row)*patch_height
      patch = img.crop(i, y_cord, i+patch_width, y_cord+patch_height)
      new_im.paste(patch, (i,y_cord))
      #print(i, y_cord)
      i = (i+patch_width) # +padding
      j += 1

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

    def __init__(self, root, split='train', transform=None, target_transform=None, blacklisted_classes=[], verbose=0,n_samples=0, pre_rotation=False, min_width=0, min_height=0, n_rows=2, n_rand_rotate=2):
        super(ROD_quarter, self).__init__(self, root, split=split, transform=transform, target_transform=target_transform, blacklisted_classes=blacklisted_classes, verbose=verbose,n_samples=n_samples, pre_rotation=pre_rotation, min_width=min_width, min_height=min_height)

        # 
        self.n_rows, self.n_rand_rotate = n_rows, n_rand_rotate
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

        self.data["patch_rotations"] = patch_label

        # Encode the relative rotation (0, 90, ...)
        le2 = preprocessing.LabelEncoder()
        self.le2 = le2
        self.data['encoded_patch_rotations'] = self.le2.fit_transform(self.data["patch_rotations"])

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (rgb_image, depth_image, enc_label, enc_patch_rotations) where target is class_index of the target class.
        '''

        rgb_image, depth_image, label, patch_rotations, enc_patch_rotations = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['patch_rotations'], self.data.iloc[index]['encoded_patch_rotations'] # Provide a way to access image and label via index

        t_depth_image =  patchize(depth_image, patch_rotations)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            #t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)            
            
        return rgb_image, depth_image, t_depth_image, label, enc_patch_rotations


class SynROD_patch(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, blacklisted_classes=[], verbose=0, n_samples=None, pre_rotation=False, min_width=0, min_height=0, n_rows=2, n_rand_rotate=2):
        
        """ Dataloader constructor.
        
        params:
        verbose: set a value >1 to have some useful infos.
        n_sample: Load a given number of sample with a random sapling (without replacement) tecnique. Reduce memory footprint.
        """
        
        super(SynROD_patch, self).__init__(self, root, split=split, transform=transform, target_transform=target_transform, blacklisted_classes=blacklisted_classes, verbose=verbose,n_samples=n_samples, pre_rotation=pre_rotation, min_width=min_width, min_height=min_height)

        self.n_rows, self.n_rand_rotate = n_rows, n_rand_rotate
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

        self.data["patch_rotations"] = patch_label

        # Encode the relative rotation (0, 90, ...)
        le2 = preprocessing.LabelEncoder()
        self.le2 = le2
        self.data['encoded_patch_rotations'] = self.le2.fit_transform(self.data["patch_rotations"])
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample,depth_image, target) where target is class_index of the target class.
        '''

        rgb_image, depth_image, label, patch_rotations, enc_patch_rotations = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['patch_rotations'], self.data.iloc[index]['encoded_patch_rotations'] # Provide a way to access image and label via index

        t_depth_image =  patchize(depth_image, patch_rotations)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            #t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)            
            
        return rgb_image, depth_image, t_depth_image, label, enc_patch_rotations
