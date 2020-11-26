import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.image as mpimg 

IMAGE_EXT = '.png'
DEFAULT_IMAGE_SHAPE = (32, 32, 3)

class ImagesLoader():

    """

    ImagesLoader: Class to load images from given folder (path as a parameter).
    
    Will load the images in a numpy array with shape: (nbr_images, image_height, image_width, nbr_channels)
    And will also create a dictionary to track the images_ids_to_index.
    Image with ID='00AF01A3HI20S28J'
    index = self.images_ids_to_index['00AF01A3HI20S28J']
    image_data = self.images_data[index]

    Parameters:
    ----------
    images_folder_path : str
        Full path of the src folder containing the images.
        
    Example:
    --------
    
    >>> images_loader = ImagesLoader(src_folder_path = 'train_profile_images')
    >>> images_loader.nbr_images
    7500
    >>> images_loader.shape
    (32, 32, 3)

    """

    def __init__(self, src_folder_path):
        self.src_folder_path = src_folder_path
        self.is_data_loaded = False
        self.images_data = None
        self.images_ids_to_index = dict()

        if not Path(src_folder_path).exists():
            err_message = 'Src folder not found! {:}'.format(src_folder_path)
            raise Exception(err_message)
        
        # Load images at initialization
        self._load_images()

    # TODO implement to read from cache file (npz)
    def _load_images_from_cache():
        raise NotImplementedError

    
    # TODO implement to save in a cache file (npz) for quicker loading from disk
    def _save_images_to_cache_file():
        raise NotImplementedError


    def _load_images(self):
        # TODO verify if a cache file is present (npz)
        # if self.cached exists...
        #    load cached instead of iterating through images in folder...

        # Compute how many images are in folder
        images_paths_list = os.listdir(self.src_folder_path)
        images_paths_list = [image_file for image_file in images_paths_list if image_file.endswith(IMAGE_EXT)]
        nbr_images_in_folder = len(images_paths_list)

        # Initiate nparray with nbr_images
        self.images_data = np.empty((nbr_images_in_folder, *DEFAULT_IMAGE_SHAPE))#, dtype=)

        # Iterate through all images and load in numpy matrix
        nbr_loaded_images = 0
        for image_index, image_file_name in enumerate(images_paths_list, start=0):
            if image_file_name.endswith(IMAGE_EXT):
                # Extract image id from file (ex: '00AF01A3HI20S28J')
                image_id = image_file_name.split('/')[-1].replace(IMAGE_EXT, '')

                # Read image from file and stock in self.images_data[]
                image_file_path = os.path.join(self.src_folder_path, image_file_name)
                try:
                    #image_data = cv2.imread(image_file_path)
                    image_data = mpimg.imread(image_file_path)
                    if image_data.shape == DEFAULT_IMAGE_SHAPE:
                        self.images_ids_to_index[image_id] = image_index
                        self.images_data[image_index] = image_data
                        nbr_loaded_images += 1

                except Exception as exp:
                    print('Failed to load image: {:}. Error: {:}'.format(image_file_name, exp))
                    continue

        # TODO: Compare nbr_loaded_images and nbr_images_in_folder to fail or succeed loading
        if nbr_loaded_images == nbr_images_in_folder:
            self.is_data_loaded = True

    # Get image data based on ID
    def get_image_data_for_profile_id(self, id_str):
        if self.is_data_loaded:
            try:
                index = self.images_ids_to_index[id_str]
                image_data = self.images_data[index]
                return image_data

            except Exception:
                return None
        else:
            return None


    @property
    def nbr_images(self):
        if self.is_data_loaded:
            return self.images_data.shape[0]
        else:
            return None

    @property
    def image_shape(self):
        if self.is_data_loaded:
            return self.images_data.shape[1:]
        else:
            return None
    



