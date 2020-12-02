import pandas as pd
import numpy as np
from pathlib import Path
import os

class TextDataLoader():

    """

    TextDataLoader: Class to load text data as DataFrame from a given csv file (path as a parameter).

    Will load the data in pandas DataFrame. Orignial data will be stored in a DataFrame and new
    encoded/transfored featrures will be stored in a new DataFrame. Both data frames can be accessed by the
    client. TODO?

    Parameters:
    ----------
    src_csv_file_path : str
        Full path of the src csv file.

    Example:
    --------

    >>> text_data_loader = TextDataLoader(src_csv_file_path = train_text_path)
    >>> text_data_loader.get_orig_data_for_profile_id(profile_id)
    AL85S14OMDPF01I9,Mf9vfld4Vfe,,Set,Verified,db1a2c,eaf0f2,e70409,False,39600.0,,Enabled,en,Thu Nov 27 05:24:59 +0000 2008,Sydney,95763,4289,30809,873,business,14.792,1.5761,AL85S14OMDPF01I9.png,2815

    """

    def __init__(self, src_csv_file_path):
        self.src_csv_file_path = src_csv_file_path
        self.is_data_loaded = False
        self.are_features_processed = False
        self.orig_data = pd.DataFrame()
        self.transformed_features = pd.DataFrame()

        if not Path(src_csv_file_path).exists():
            err_message = 'Src csv file not found! {:}'.format(src_csv_file_path)
            raise Exception(err_message)

        # Load images at initialization
        self._load_text_data_from_file()


    def _load_text_data_from_file(self):
        self.orig_data = pd.read_csv(self.src_csv_file_path)
        self.is_data_loaded = True

 
    # Return orignal data (entire dataframe)
    def get_orig_features(self):
        if self.is_data_loaded:
            return self.orig_data
        return None


    # Return original data for only the provided id
    def get_orig_features_for_profile_id(self, id):
        if self.is_data_loaded:
            return self.orig_data[self.orig_data['Id'] == id]
        return None
