# AI Programming with Python Project

## Description
Project code for Udacity's AI Programming with Python Nanodegree program. The project is to develop using PyTorch an image classifier to recognize image of flowers, using the dataset from [https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). 


The project contains a Jupyter Notebook and a command line project based on two scripts, train.py and predict.py
train.py loads the data set, initializes and trains a deep neural network, then saves the model as a checkpoint.

Basic usage : <code>python train.py data_directory</code>
predict.py loads an image and a previously trained model from a checkpoint, then returns the flower name and class probability Basic usage : <code>python predict.py /path/to/image checkpoint</code>

## Libraries
 - PyTorch
 - Numpy
 - Pillow
