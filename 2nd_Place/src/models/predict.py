import sys
from loguru import logger
import numpy as np
import typer
from tifffile import imwrite
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import os

import tensorflow as tf
from tensorflow import keras
import rasterio

import keras.backend as K

def parent(path_):
    return os.path.abspath(os.path.join(path_, os.pardir))

def IOU_coef(y_true, y_pred):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(y_true_f * y_pred_f)
  return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
def DiceLoss_square(y_true, y_pred, smooth=1):
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred)
  intersection = K.sum(K.abs(y_true_f * y_pred_f))
  return 1-((2. * intersection + smooth) / (K.sum(K.square(y_true_f),-1) + K.sum(K.square(y_pred_f),-1) + smooth))

def DiceLoss(y_true, y_pred, smooth=1):
  intersection = K.sum(y_true * y_pred)
  return 1-((2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth))


def make_predictions(chip_id: str, models):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """
    #logger.info(os.path.join(os.getcwd(), 'data', 'test_features', chip_id+'_vh.tif'))
    try:

        dirs = ["test_features", "test_features", "nasadem", "jrc_change", "jrc_extent", "jrc_seasonality", "jrc_occurrence", "jrc_recurrence", "jrc_transitions"]
        endings = ["_vv.tif", "_vh.tif", ".tif", ".tif", ".tif", ".tif", ".tif", ".tif", ".tif"]
        
        arrays = []

        #os.path.abspath()
        for i in range(9):
            path = os.path.join(parent(parent(os.getcwd())), 'data', 'to_predict', dirs[i], chip_id + endings[i])

            #print(path)
            #load image from path
            with rasterio.open(path) as img:
        
                if(i < 2):
                    arrays.append(np.uint8(np.clip(img.read(1), -30, 0)*(-8.4)))
                elif(i == 2):
                    arrays.append(np.uint8(np.clip(img.read(1), 0, 255)))
                else:
                    arrays.append(img.read(1))
        
        images = np.array([np.stack(arrays, axis=-1)])

        #logger.info(img.shape)

        #config = model.get_config() # Returns pretty much every information about your model
        #logger.info(config["layers"][0]["config"]["batch_input_shape"]) # returns a tuple of width, height and channels
        output_predictions = []

        for model in models:
            output_predictions.append(model.predict(images)[0,:, :, 0])
        
        output_prediction =  np.mean(output_predictions, axis=0)
        output_prediction = ((output_prediction > 0.5) * 1).astype(np.uint8)

        #logger.info(output_prediction.shape)

    except:
        logger.warning(
            f"test_features not found for {chip_id}, predicting all zeros; did you download your"
            f"training data into `runtime/data/test_features` so you can test your code?"
        )
        output_prediction = np.zeros(shape=(512, 512))
    return output_prediction


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    mypath = os.path.join(parent(parent(os.getcwd())), 'data', 'to_predict', 'test_features')
    files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files.remove(".gitkeep")
    
    #logger.info(files)
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(f.split("_")[0] for f in files)))
    #logger.info(ids)
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    logger.info("Loading model")
    custom_objects = {"DiceLoss": DiceLoss, "DiceLoss_square": DiceLoss_square, "IOU_coef": IOU_coef}
    models = []
    with keras.utils.custom_object_scope(custom_objects):
        models.append(keras.models.load_model(os.path.join(parent(parent(os.getcwd())), 'models', 'model_floodwater_unet_pc_augm_diceloss.h5')))
        models.append(keras.models.load_model(os.path.join(parent(parent(os.getcwd())), 'models', 'model_floodwater_unet_pc_augm_diceloss_2.h5')))
        models.append(keras.models.load_model(os.path.join(parent(parent(os.getcwd())), 'models', 'model_floodwater_unet_pc_augm_diceloss_3.h5')))
        
    #logger.info(model.summary())


    logger.info("Finding chip IDs in ")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")
    for chip_id in tqdm(chip_ids, miniters=25, file=sys.stdout, leave=True):
        # figure out where this prediction data should go
        output_path = os.path.join(parent(parent(os.getcwd())), 'output_data', chip_id+'.tif')
        # make our predictions! (you should edit `make_predictions` to do something useful)
        output_data = make_predictions(chip_id, models)
        imwrite(output_path, output_data, dtype=np.uint8)
    logger.success(f"... done")


if __name__ == "__main__":
    typer.run(main)
