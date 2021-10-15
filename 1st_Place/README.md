# STAC-Overflow
1st place solution for STAC Overflow: Map Floodwater from Radar Imagery hosted by Microsoft AI for Earth
https://www.drivendata.org/competitions/81/detect-flood-water/

If this solution seemed useful to you, be sure to share ⭐️


## About the solution

Initially, I understood that I would not be able to build a super complex neural network, because either there was not enough knowledge or there was not enough computing power.

Therefore, the only chance to win was to come up with a simpler method for determining flooding. To do this, I studied articles about how waterlogging is determined now. There were neural network methods, but there were also mathematical methods. From which I concluded that in addition to segmentation by a neural network, you can try to determine the flooding pixel by pixel by some formula.

But since I am a "cool" data scientist 🦧, I did not output the formula manually, but trained ML models – Catboostclassifier, which solved the binary classification problem on pixel-by-pixel data.

Before that, I also trained the Unet models.

Further, I noticed that the models often do not fill the necessary zones, rather than overfill. Therefore, I combined the predictions of these two approaches, taking their maxima, not the average.

And as you can see, this approach worked and brought me such an important victory! 🥳

You can see other notes about the solution in the jupyter-notebooks.


## Solution

This solution assumes that training features are saved in the directory `../training_data/train_features`, training labels are saved in the directory `../training_data/train_labels`, and the metadata is saved to `../training_data/flood-training-metadata.csv`.

1. __load_external_data.ipynb__ 

This notebook is downloading additional data from Planetary Computer. Spoiler: Nasadem band is an incredibly important


2. __catboost_model.ipynb__ 

This shows the preparation of pixel-by-pixel data and the training of CatBoostClassifier models on them.


3. __unet_model.ipynb__ 

Here is a classic segmentation approach using neural networks with the Unet architecture with EfficientNet backbone.


4. __compare_methods.ipynb__ 

This notebook shows a comparison of the results of the two approaches and their combination.

5. __inference.py__

This script performs inference on the test set using the saved model weights.
