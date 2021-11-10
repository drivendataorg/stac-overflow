import os
from pathlib import Path

from loguru import logger
import numpy as np
from tifffile import imwrite
from tqdm import tqdm
import torch
import typer

from flood_model import FloodModel


ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
ASSETS_DIRECTORY = ROOT_DIRECTORY / "assets"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"
CHANGE  = DATA_DIRECTORY / "jrc_change"
EXTENT  = DATA_DIRECTORY / "jrc_extent"
OCCUR   = DATA_DIRECTORY / "jrc_occurrence"
RECURR  = DATA_DIRECTORY / "jrc_recurrence"
SEAS    = DATA_DIRECTORY / "jrc_seasonality"
TRANS   = DATA_DIRECTORY / "jrc_transitions"
NASADEM = DATA_DIRECTORY / "nasadem"

# Make sure the smp loader can find our torch assets because we don't have internet!
os.environ["TORCH_HOME"] = str(ASSETS_DIRECTORY / "torch")


def make_prediction(chip_id, model):
    """
    Given a chip_id, read in the vv/vh bands and predict a water mask.

    Args:
        chip_id (str): test chip id

    Returns:
        output_prediction (arr): prediction as a numpy array
    """
    logger.info("Starting inference.")
    try:
        output_prediction = None
        for e,m in enumerate(model):
            vv_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif"
            vh_path = INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif"
            change_path  = CHANGE / f"{chip_id}.tif"
            extent_path  = EXTENT / f"{chip_id}.tif"
            occur_path   = OCCUR / f"{chip_id}.tif"
            recurr_path  = RECURR / f"{chip_id}.tif"
            seas_path    = SEAS / f"{chip_id}.tif"
            trans_path   = TRANS / f"{chip_id}.tif"
            nasadem_path = NASADEM / f"{chip_id}.tif"
            if e == 0:
                output_prediction = m.predict(vv_path, vh_path, change_path, extent_path, occur_path, recurr_path, seas_path, trans_path, nasadem_path)
            else:
                output_prediction += m.predict(vv_path, vh_path, change_path, extent_path, occur_path, recurr_path, seas_path, trans_path, nasadem_path)
                
    except Exception as e:
        logger.error(f"No bands found for {chip_id}. {e}")
        raise
    return output_prediction


def get_expected_chip_ids():
    """
    Use the test features directory to see which images are expected.
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # Return one chip id per two bands (VV/VH)
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    For each set of two input bands, generate an output file
    using the `make_predictions` function.
    """
    logger.info("Loading model")
    # Explicitly set where we expect smp to load the saved resnet from just to be sure
    torch.hub.set_dir(ASSETS_DIRECTORY / "torch/hub")
    folds = 5
    models = []
    for i in range(folds):
        model = FloodModel()
        model.load_state_dict(torch.load(ASSETS_DIRECTORY / "flood_model-{}.pt".format(i)))
        models.append(model.cuda())

    logger.info("Finding chip IDs")
    chip_ids = get_expected_chip_ids()
    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(chip_ids)} test chip_ids. Generating predictions.")
    for chip_id in tqdm(chip_ids, miniters=25):
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        output_data = make_prediction(chip_id, models).astype(np.uint8)
        imwrite(output_path, ((output_data > 2.5) * 1).astype(np.uint8), dtype=np.uint8)

    logger.success(f"Inference complete.")

if __name__ == "__main__":
    typer.run(main)
