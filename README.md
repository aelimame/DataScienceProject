# DataScienceProject

train.py : To use to train the models

model.py :  NN models

utils folder : Folder for utilities, data loaders...

Notebook-Experiences.ipynb: Jupyter notebook for experiments and analysis

Work in progress!

## Install

For Mac OSX: brew install libomp

## Troubleshooting

If you had issue related to AsyncGeneratorItem when connecting your conda environment to jupyter use:

pip uninstall -y ipython prompt_toolkit
pip install ipython prompt_toolkit

To connect your environment:

python -m ipykernel install --user --name=<name_of_env>