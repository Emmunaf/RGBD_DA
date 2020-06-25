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

from google.colab import drive


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class ROD(VisionDataset):

    def __init__(self, root, split='train', transform=None, target_transform=None, blacklisted_classes=[], verbose=0,n_samples=0, pre_rotation=False, min_width=0, min_height=0):
        super(ROD, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.pre_rotation = pre_rotation
        #self.blacklist_classes = blacklisted_classes  # Needed just is using custom mapper
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        '''
        
        split_path = os.path.join(root, split+".txt")  # 
        split_file = np.loadtxt(split_path, dtype='str')
        
        skipped_minh, skipped_minw = 0, 0
        imgs_and_labels = []
        missing_couple = 0
        seed(42)
        if n_samples is not None and n_samples > 0:
            #split_file = sample(split_file, n_samples)
            split_file = split_file[np.random.choice(split_file.shape[0], n_samples, replace=False)]
        for line in split_file:
          image_path, classN = line
          class_label = image_path.split('/')[1]
          if class_label in blacklisted_classes:
            if verbose > 0:
              print("[INFO] Skipping 1 class because of blacklist param. Skipped class: ", class_label)
            continue
          
          rgb_img_path = root+image_path.replace("???", "rgb").replace("***","crop")
          depth_img_path = root+image_path.replace("???", "surfnorm").replace("***","depthcrop")
          # Skip if one of the 2 (domains) images is not where it should be
          if not os.path.isfile(depth_img_path) or not os.path.isfile(rgb_img_path) :
              missing_couple += 1
              if verbose > 0:
                print("[INFO] Skipping 1 sample because of missing partner")
              continue
          
          # Load the image
          depth_img = pil_loader(depth_img_path)
          rgb_img = pil_loader(rgb_img_path)
          rwidth, rheight = rgb_img.size
          dwidth, dheight = depth_img.size
        
          # Apply min_width and min_height filter if the arg was set
          if min_width > 0 and (rwidth <= min_width or dwidth <= min_width):
                skipped_minw += 1
                if min_height > 0 and (rheight <= min_height or dheight <= min_height):
                    skipped_minh += 1
                continue
          if min_height > 0 and (rheight <= min_height or dheight <= min_height):
                skipped_minh += 1
                continue

          # Generate random rotations and save the delta
          # NOTE: As of now assume that the depth  and rgb images have always the same rotation applied
          depth_rotation = randint(0, 3) * 90
          rgb_rotation = randint(0, 3) * 90
          relative_rotation = depth_rotation - rgb_rotation
          if relative_rotation < 0:
            relative_rotation += 360

          if pre_rotation:
            t_rgb_img = rgb_img.rotate(rgb_rotation)
            t_depth_img = depth_img.rotate(depth_rotation)
            imgs_and_labels.append([rgb_img, depth_img, t_rgb_img, t_depth_img, class_label, relative_rotation])  # add rgb/depth rotation?
          else:
            imgs_and_labels.append([rgb_img, depth_img, rgb_rotation, depth_rotation, class_label, relative_rotation])  # add rgb/depth rotation?

        if min_height > 0 or min_width > 0:
          print("[INFO] A total of ", skipped_minw, " samples were skipped because of of min_width parameter")
          print("[INFO] A total of ", skipped_minh, " samples were skipped because of of min_height parameter")
        print("[INFO] A total of ", missing_couple, "samples were skipped because of a missing partner domain")


        self.data = pd.DataFrame(imgs_and_labels, columns=['rgb', 'depth','t_rgb', 't_depth', 'class', 'relative_rotation'])
        del(imgs_and_labels)
        gc.collect()
        # USing custom enc, notthe txt given one
        le = preprocessing.LabelEncoder()
        self.le = le
        self.data['encoded_class'] = self.le.fit_transform(self.data['class'])
        
        # Encode the relative rotation (0, 90, ...)
        le2 = preprocessing.LabelEncoder()
        self.le2 = le2
        self.data['encoded_relative_rot'] = self.le2.fit_transform(self.data['relative_rotation'])

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample, depth_image, target) where target is class_index of the target class.
        '''
        if self.pre_rotation:
            rgb_image, depth_image,t_rgb_image, t_depth_image, label, encoded_relative_rot = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'],self.data.iloc[index]['t_rgb'], self.data.iloc[index]['t_depth'], self.data.iloc[index]['encoded_class'],  self.data.iloc[index]['encoded_relative_rot'] # Provide a way to access image and label via index
        else:
            rgb_image, depth_image, rgb_rotation, depth_rotation, label, encoded_relative_rot = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'],self.data.iloc[index]['t_rgb'], self.data.iloc[index]['t_depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['encoded_relative_rot'] # Provide a way to access image and label via index
            t_rgb_image, t_depth_image = rgb_image.rotate(rgb_rotation), depth_image.rotate(depth_rotation)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)            
            
        return rgb_image, depth_image,t_rgb_image, t_depth_image, label, encoded_relative_rot

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
    
    def split_data(self, val_size=0.5):
        """
        Split the train set in to train and validation set (stratified sampling)
        
        args:
            val_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include for validation
        returns:
            (train_indexes[], val_indexes[]): lists of indexes for train and validation split.
        """

        X_train, X_val = train_test_split(self.data, test_size=val_size, stratify=self.data['encoded_class'] )
    
        # Get (not contiguous) indexes for a stratified split 
        train_indexes, val_indexes = X_train.index.values, X_val.index.values

        # Create an ordered dataframe to have contiguous index ranges (ie. 0-2000, 2000-4000)
        new_train_dataset = self.data.filter(train_indexes, axis=0)
        new_val_dataset = self.data.filter(val_indexes, axis=0)
        new_dataset = pd.DataFrame(new_train_dataset).reset_index(drop=True)
        new_dataset = new_dataset.append(new_val_dataset, ignore_index=True)
        # Assign new dataframe to data attribute
        self.data = new_dataset
        # Define the contiguous indexes by using just length
        train_indexes, val_indexes = list(range(len(train_indexes))), list(range(len(train_indexes), len(train_indexes)+len(X_val.index.values)))

        return train_indexes, val_indexes
    
    def reduce_data(self, tokeep=0.6):
        """
        Reduce the entire set using a stratified sampling
        
        args:
            tokeep: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
        returns:
            (Int, int): Indexes of remaining elements (contigous), Num of deleted elements.
        """

        X_train, X_val = train_test_split(self.data, test_size=tokeep, stratify=self.data['encoded_class'] )
    
        # Get (not contiguous) indexes for a stratified split 
        del_indexes, new_indexes = X_train.index.values, X_val.index.values

        # Create an ordered dataframe to have contiguous index ranges (ie. 0-2000)
        new_val_dataset = self.data.filter(new_indexes, axis=0)
        new_dataset = pd.DataFrame(new_val_dataset).reset_index(drop=True)
        # Assign new dataframe to the cutted new dataset
        self.data = new_dataset

        gc.collect()  # Force garbace collector 
        # Define the contiguous indexes by using just length
        new_indexes, del_indexes = list(range(len(new_indexes))), list(range(len(del_indexes), len(new_indexes)+len(del_indexes)))
        return new_indexes, len(del_indexes)
    
    def get_classes(self):
        """Return the classes as list """
        
        return self.le.classes_#self.data['class']
    
    def get_encoded_classes(self):
        """Return the ecoded classes mapping dict"""
        
        class_mapping = {self.le.transform(v):v for v in self.le.classes_}
        return class_mapping

class SynROD(VisionDataset):

    def __init__(self, root, transform=None, target_transform=None, blacklisted_classes=[], verbose=0, n_samples=None, pre_rotation=False, min_width=0, min_height=0):
        
        """ Dataloader constructor.
        
        params:
        verbose: set a value >1 to have some useful infos.
        n_sample: Load a given number of sample with a random sapling (without replacement) tecnique. Reduce memory footprint.
        """
        super(SynROD, self).__init__(root, transform=transform, target_transform=target_transform)
        self.pre_rotation = pre_rotation
            
        #self.blacklist_classes = blacklisted_classes  # Needed just is using custom mapper
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        imgs_path = []
        imgs_and_labels = []
        missing_couple = 0
        
        seed(42)
        
        parent, dirs, files = next(os.walk(root))
        for dir_name in dirs:  # Iterate over class-named folder (apple, ball, banana)
          class_label = dir_name
          depth_folder_path  = os.path.join(root, dir_name, "depth") 
          rgb_folder_path  = os.path.join(root, dir_name, "rgb") 
          if class_label in blacklisted_classes:
            if verbose > 0:
              print("[INFO] Skipping 1 class because of blacklist param. Skipped class: ", class_label)
            continue
          _, _, imgs = next(os.walk(rgb_folder_path))
          for img in imgs:
            rgb_img_path  = os.path.join(rgb_folder_path, img) 
            depth_img_path  = os.path.join(depth_folder_path, img) 
            if not os.path.isfile(depth_img_path):
              missing_couple += 1
              if verbose > 0:
                print("[INFO] Skipping 1 sample because of missing depth partner")
                print(depth_img_path, "doesnt exist")
              continue
              # raise Exception("For each rgb image you NEED the depth version too")
            imgs_path.append([rgb_img_path, depth_img_path, class_label])  # add rgb/depth rotation?
        
        # IF n_samples was set load only a given number of sample (random and without replacement)
        prepruned = len(imgs_path)
        if n_samples is not None and n_samples > 0:
            imgs_path = sample(imgs_path, n_samples)
        
        skipped_minh, skipped_minw = 0, 0
        for rgb_img_path, depth_img_path, class_label in imgs_path:    
          # Load the image
          depth_img = pil_loader(depth_img_path)
          rgb_img = pil_loader(rgb_img_path)
          rwidth, rheight = rgb_img.size
          dwidth, dheight = depth_img.size
        
          # Apply min_width and min_height filter if the arg was set
          if min_width > 0 and (rwidth <= min_width or dwidth <= min_width):
                skipped_minw += 1
                if min_height > 0 and (rheight <= min_height or dheight <= min_height):
                    skipped_minh += 1
                continue
          if min_height > 0 and (rheight <= min_height or dheight <= min_height):
                skipped_minh += 1
                continue
          
          # Generate random rotations and save the delta
          # NOTE: As of now assume that the depth  and rgb images have always the same rotation applied
          depth_rotation = randint(0, 3) * 90
          rgb_rotation = randint(0, 3) * 90
          relative_rotation = depth_rotation - rgb_rotation
          if relative_rotation < 0:
            relative_rotation += 360

          if pre_rotation:
              t_rgb_img, t_depth_img = rgb_img.rotate(rgb_rotation), depth_img.rotate(depth_rotation)
              # Note: Can be better to reduce in a way that resolve the unbalancing of classes?
              # As of now this is *NOT* taken into account
              imgs_and_labels.append([rgb_img, depth_img, t_rgb_img, t_depth_img, class_label, relative_rotation])  # add rgb/depth rotation?
              prepruned -= len(imgs_and_labels) 
              print("[INFO] A total of ", prepruned, "samples were skipped because of the pre_prune_ratio parameters")
          else:
              imgs_and_labels.append([rgb_img, depth_img, rgb_rotation, depth_rotation, class_label, relative_rotation])  # add rgb/depth rotation?

        del(imgs_path)
        if min_height > 0 or min_width > 0:
          print("[INFO] A total of ", skipped_minw, " samples were skipped because of of min_width parameter")
          print("[INFO] A total of ", skipped_minh, " samples were skipped because of of min_height parameter")
        print("[INFO] A total of ", missing_couple, "samples were skipped because their depth map it's missing")

        self.data = pd.DataFrame(imgs_and_labels, columns=['rgb', 'depth','t_rgb', 't_depth', 'class', 'relative_rotation'])
        
        # Note: Using custom enc, not the txt given one
        le = preprocessing.LabelEncoder()
        self.le = le
        self.data['encoded_class'] = self.le.fit_transform(self.data['class'])       
        # Encode relative rotation
        le2 = preprocessing.LabelEncoder()
        self.le2 = le2
        self.data['encoded_relative_rot'] = self.le2.fit_transform(self.data['relative_rotation'])
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index
        Returns:
            tuple: (sample,depth_image, target) where target is class_index of the target class.
        '''
        if self.pre_rotation:
            rgb_image, depth_image, t_rgb_image, t_depth_image, label, encoded_relative_rot = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'],self.data.iloc[index]['t_rgb'], self.data.iloc[index]['t_depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['encoded_relative_rot'] # Provide a way to access image and label via index
        else:
            rgb_image, depth_image, rgb_rotation, depth_rotation, label, encoded_relative_rot = self.data.iloc[index]['rgb'], self.data.iloc[index]['depth'],self.data.iloc[index]['t_rgb'], self.data.iloc[index]['t_depth'], self.data.iloc[index]['encoded_class'], self.data.iloc[index]['encoded_relative_rot'] # Provide a way to access image and label via index
            t_rgb_image, t_depth_image = rgb_image.rotate(rgb_rotation), depth_image.rotate(depth_rotation)
        # Applies preprocessing when accessing the image
        if self.transform is not None:
            rgb_image = self.transform(rgb_image)
            t_rgb_image = self.transform(t_rgb_image)
            depth_image = self.transform(depth_image)
            t_depth_image = self.transform(t_depth_image)
            
        return rgb_image, depth_image,t_rgb_image, t_depth_image, label, encoded_relative_rot

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.data) # Provide a way to get the length (number of elements) of the dataset
        return length
    
    def split_data(self, val_size=0.5):
        """
        Split the train set in to train and validation set (stratified sampling)
        
        args:
            val_size: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include for validation
        returns:
            (train_indexes[], val_indexes[]): lists of indexes for train and validation split.
        """

        X_train, X_val = train_test_split(self.data, test_size=val_size, stratify=self.data['encoded_class'] )
    
        # Get (not contiguous) indexes for a stratified split 
        train_indexes, val_indexes = X_train.index.values, X_val.index.values

        # Create an ordered dataframe to have contiguous index ranges (ie. 0-2000, 2000-4000)
        new_train_dataset = self.data.filter(train_indexes, axis=0)
        new_val_dataset = self.data.filter(val_indexes, axis=0)
        new_dataset = pd.DataFrame(new_train_dataset).reset_index(drop=True)
        new_dataset = new_dataset.append(new_val_dataset, ignore_index=True)
        # Assign new dataframe to data attribute
        self.data = new_dataset
        # Define the contiguous indexes by using just length
        train_indexes, val_indexes = list(range(len(train_indexes))), list(range(len(train_indexes), len(train_indexes)+len(X_val.index.values)))

        return train_indexes, val_indexes

    def reduce_data(self, tokeep=0.6):
        """
        Reduce the entire set using a stratified sampling
        
        args:
            tokeep: If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include
        returns:
            (Int, int): Indexes of remaining elements (contigous), Num of deleted elements.
        """

        X_train, X_val = train_test_split(self.data, test_size=tokeep, stratify=self.data['encoded_class'] )
    
        # Get (not contiguous) indexes for a stratified split 
        del_indexes, new_indexes = X_train.index.values, X_val.index.values

        # Create an ordered dataframe to have contiguous index ranges (ie. 0-2000)
        new_val_dataset = self.data.filter(new_indexes, axis=0)
        new_dataset = pd.DataFrame(new_val_dataset).reset_index(drop=True)
        # Assign new dataframe to the cutted new dataset
        self.data = new_dataset

        gc.collect()  # Force garbace collector 
        # Define the contiguous indexes by using just length
        new_indexes, del_indexes = list(range(len(new_indexes))), list(range(len(del_indexes), len(new_indexes)+len(del_indexes)))
        return new_indexes, len(del_indexes)
    
    def get_classes(self):
        """Return the classes as list """
        
        return self.le.classes_#self.data['class']
    
    def get_encoded_classes(self):
        """Return the ecoded classes mapping dict"""
        
        class_mapping = {self.le.transform(v):v for v in self.le.classes_}
        return class_mapping
    
