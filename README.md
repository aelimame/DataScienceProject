# DataScienceProject

## Porject structure and files

train_predict.py : Use to train on Train set (train.csv) and predict on the Test set (test.csv) using simple models (Random Forest, Gradient Boost Regressor...). This uses text featrues only.

train_dlnn.py: Use this to train on text features (train.csv) and images (train_profile_images). This uses Keras DL NN models (see keras_models.py script).

keras_models.py: Keras models to train on text features and images

utils : Folder for utilities, data loaders, data transformer...

utils\text_data_loader.py: Helper class to load features from csv files.

utils\images_loader.py: Helper class to load images from folders. It helps with matching images with profile ids.

utils\text_data_transformer.py: Helper class to transform features.

Notebook-Experiences.ipynb: Jupyter notebook for experiments and analysis

submissions: Folder containing the preditions csv files (on the test set) submitted to Kaggle

submissions\submissions_info.csv: Contains all the information about the submitted predictions (git code version, summary about the models used, score we got on Kaggle...).

Work in progress!

## IMPORTANT: Predictions submission
Use train_predict.py to generate predictions on the Test set (you can try new models...).
Before submitting the predictions, make sure to push the new/modified code to github to keep track of the code used to make the predictions. This is very important so we can reproduce the predictions and track the models used. Make sure to update the sumbmission csv file in submissions folder, add comments about the model(s) used, the features used, the new transformations done to the data and any other relevent information. Don't forget to add the prediction file itself to the subfolder submissions\pred_files. Also, name the prediction file based on the model, date, git version...


## Install

Use requirements.txt to install all needed libs (pip install -r requirements.txt). Highly recommend using mini-conda envirements.

For Mac OSX: brew install libomp


## Troubleshooting

If you had issue related to AsyncGeneratorItem when connecting your conda environment to jupyter use:

pip uninstall -y ipython prompt_toolkit
pip install ipython prompt_toolkit

To connect your environment:

python -m ipykernel install --user --name=<name_of_env>