import os
from pathlib import Path

from catboost import CatBoostClassifier
from loguru import logger
import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from tifffile import imwrite, imsave
from tqdm import tqdm
import typer


SUBMISSION_DIRECTORY = Path("submission")
ASSETS_DIRECTORY = Path("assets")
INPUT_IMAGES_DIRECTORY = Path("data/test_features")
NASADEM_DIRECTORY = Path('data/nasadem')
JRC_CHANGE_DIRECTORY = Path('data/jrc_change')
JRC_OCCURANCE_DIRECTORY = Path('data/jrc_occurrence')
JRC_EXTENT_DIRECTORY = Path('data/jrc_extent')
JRC_RECURRENCE_DIRECTORY = Path('data/jrc_recurrence')
JRC_SEASONALITY_DIRECTORY = Path('data/jrc_seasonality')
JRC_TRANSITIONS_DIRECTORY = Path('data/jrc_transitions')


def bce_jaccard_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return (1 - jac) * smooth + tf.keras.losses.binary_crossentropy(y_true, y_pred)


def make_prediction(chip_id, models_nn_1, models_nn_2, models_cat):
    
    logger.info("Starting inference.")

    try:
        vv_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif"
        vh_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif"
        nasadem_path = NASADEM_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_change_path = JRC_CHANGE_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_occurrence_path = JRC_OCCURANCE_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_extent_path = JRC_EXTENT_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_recurrence_path = JRC_RECURRENCE_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_seasonality_path = JRC_SEASONALITY_DIRECTORY / f"{chip_id}.tif"
        jrc_gsw_transitions_path = JRC_TRANSITIONS_DIRECTORY / f"{chip_id}.tif"

        with rasterio.open(vv_path) as fvv:
            vv = fvv.read(1)
        with rasterio.open(vh_path) as fvh:
            vh = fvh.read(1)
        with rasterio.open(nasadem_path) as fnasadem:
            nasadem = fnasadem.read(1)
        with rasterio.open(jrc_gsw_change_path) as fjrc_gsw_change:
            jrc_gsw_change = fjrc_gsw_change.read(1)
        with rasterio.open(jrc_gsw_occurrence_path) as fjrc_gsw_occurrence:
            jrc_gsw_occurrence = fjrc_gsw_occurrence.read(1)
        with rasterio.open(jrc_gsw_extent_path) as fjrc_gsw_extent:
            jrc_gsw_extent = fjrc_gsw_extent.read(1)
        with rasterio.open(jrc_gsw_recurrence_path) as fjrc_gsw_recurrence:
            jrc_gsw_recurrence = fjrc_gsw_recurrence.read(1)
        with rasterio.open(jrc_gsw_seasonality_path) as fjrc_gsw_seasonality:
            jrc_gsw_seasonality = fjrc_gsw_seasonality.read(1)
        with rasterio.open(jrc_gsw_transitions_path) as fjrc_gsw_transitions:
            jrc_gsw_transitions = fjrc_gsw_transitions.read(1)

        X = np.zeros((512, 512, 3))
        X[:, :, 0] = (vh - (-17.54)) / 5.15
        X[:, :, 1] = (vv - (-10.68)) / 4.62
        X[:, :, 2] = (nasadem - (166.47)) / 178.47

        temp = pd.DataFrame()
        temp['vh'] = vh.flatten()
        temp['vv'] = vv.flatten()
        temp['nasadem'] = nasadem.flatten()
        temp['jrc_gsw_change'] = jrc_gsw_change.flatten()
        temp['jrc_gsw_occurrence'] = jrc_gsw_occurrence.flatten()
        temp['jrc_gsw_extent'] = jrc_gsw_extent.flatten()
        temp['jrc_gsw_recurrence'] = jrc_gsw_recurrence.flatten()
        temp['jrc_gsw_seasonality'] = jrc_gsw_seasonality.flatten()
        temp['jrc_gsw_transitions'] = jrc_gsw_transitions.flatten()

        pred_cat = np.zeros((temp.shape[0], 20))
        for i in range(20):
            pred_cat[:, i] = models_cat[i].predict_proba(temp)[:, 1]
            
        pred_cat = np.mean(pred_cat, axis=1).reshape(512, 512)
        
        
        pred_nn_1 = models_nn_1[0].predict(X[np.newaxis, :, :, :])[0, :, :, 0]
        for i in range(1, 5):
            pred_nn_1 += models_nn_1[i].predict(X[np.newaxis, :, :, :])[0, :, :, 0]
        pred_nn_1 /= 5
        
        pred_nn_2 = models_nn_2[0].predict(X[np.newaxis, :, :, :])[0, :, :, 0]
        pred_nn_2 += models_nn_2[1].predict(X[np.newaxis, :, :, :])[0, :, :, 0]
        pred_nn_2 /= 2
            
        pred_all = np.max([pred_nn_1, pred_nn_2, pred_cat], axis=0)

        pred_thresh = pred_all.copy()
        pred_thresh[pred_thresh < 0.5] = 0
        pred_thresh[pred_thresh >= 0.5] = 1
        pred_thresh = pred_thresh.astype(int)

    except Exception as e:
        logger.error(f"No bands found for {chip_id}. {e}")
        raise

    return pred_thresh


def get_expected_chip_ids():
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # Return one chip id per two bands (VV/VH)
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    logger.info("Loading model")

    models_nn_1 = []
    for i in range(5):
        model = load_model(ASSETS_DIRECTORY / 'EfficientB4Unet_512_3_{}.h5'.format(i), 
                           custom_objects={'FixedDropout': Dropout,
                                           'bce_jaccard_loss': bce_jaccard_loss})
        models_nn_1.append(model)
    
    models_nn_2 = []
    model = load_model(ASSETS_DIRECTORY / 'EffUnetB0_512_3_weak_1.h5', 
                       custom_objects={'FixedDropout': Dropout,
                                       'bce_jaccard_loss': bce_jaccard_loss})
    models_nn_2.append(model)
    model = load_model(ASSETS_DIRECTORY / 'EffUnetB0_512_3.h5', 
                       custom_objects={'FixedDropout': Dropout,
                                       'bce_jaccard_loss': bce_jaccard_loss})
    models_nn_2.append(model)
    
    models_cat = []
    for i in range(4):
        model = CatBoostClassifier()
        model.load_model(ASSETS_DIRECTORY / "stratifiedkfold{}".format(i))
        models_cat.append(model)
        
    for i in range(4):
        model = CatBoostClassifier()
        model.load_model(ASSETS_DIRECTORY / "kfold{}".format(i))
        models_cat.append(model)
        
    for i in range(8):
        model = CatBoostClassifier()
        model.load_model(ASSETS_DIRECTORY / "model{}".format(i))
        models_cat.append(model)
        
    for i in range(4):
        model = CatBoostClassifier()
        model.load_model(ASSETS_DIRECTORY / "region{}".format(i))
        models_cat.append(model)

    logger.info("Finding chip IDs")

    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(chip_ids)} test chip_ids. Generating predictions.")
    for chip_id in tqdm(chip_ids, miniters=25):
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        output_data = make_prediction(chip_id,  models_nn_1, models_nn_2, models_cat).astype(np.uint8)
        imsave(output_path, output_data)

    logger.success(f"Inference complete.")


if __name__ == "__main__":
    typer.run(main)