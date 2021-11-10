# STAC Overflow

Third place solution for the STAC Overflow: Map Floodwater from Radar Imagery competition.

## Prerequisites

Install dependencies by running:

`pip install requirements.txt`

## Model training and inference

The notebook provided should run end-to-end to reproduce the trained model and setup the submission directory for prediction of the test set. This solution assumes that training features are saved in the directory `../training_data/train_features`, training labels are saved in the directory `../training_data/train_labels`, and the metadata is saved to `../training_data/flood-training-metadata.csv`.