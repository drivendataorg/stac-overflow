[<img src='https://s3.amazonaws.com/drivendata-public-assets/logo-white-blue.png' width='600'>](https://www.drivendata.org/)
<br><br>

<img src='https://drivendata-public-assets.s3.amazonaws.com/c2s-sentinel-1.jpeg' width='100%' height='300'>


# STAC Overflow: Map Floodwater from Radar Imagery

## Goal of the Competition

Flooding is the most frequent and costly natural disaster in the world. According to the World Health Organization, between 1998 and 2017, floods affected more than two billion people worldwide. As global warming continues to exacerbate rising sea levels, prolong rainstorms, and increase snowmelt, the frequency and severity of extreme weather is only expected to rise.

During a flood event, it is critical that humanitarian organizations be able to accurately measure flood extent in near real-time to strengthen early warning systems and target relief. Historically, hydrologists have relied on readings from rain and stream gauging stations to understand flood reach. Though helpful, these ground measures only measure water height, are spatially limited, and can be expensive to maintain. High resolution synthetic-aperture radar (SAR) imaging has strengthened monitoring systems by providing data in otherwise inaccessible areas at frequent time intervals. Specifically, C-band SAR from the Sentinel-1 mission provides an all-weather, day-and-night supply of images of the Earth’s surface.

The goal of this challenge is to build machine learning algorithms that are able to map floodwater using Sentinel-1 global SAR imagery. Microsoft AI for Earth has teamed up with DrivenData and Cloud to Street to investigate the applicability of machine learning models for detecting flood coverage in near real-time. Models that can effectively use remote sensing to improve flood mapping have the potential to strengthen flood risk assessment, relief targeting, and disaster preparedness.

## What's in this Repository

This repository contains code from winning competitors in the [STAC Overflow: Map Floodwater from Radar Imagery](https://www.drivendata.org/competitions/81/detect-flood-water/) DrivenData challenge.

**Winning code for other DrivenData competitions is available in the [competition-winners repository](https://github.com/drivendataorg/competition-winners).**

## Winning Submissions

Place | Team or User | Public Score | Private Score | Summary of Model
--- | --- | ---   | ---   | ---
1   | Moscow Hares: [sweetlhare](https://www.drivendata.org/users/sweetlhare/), [Belass](https://www.drivendata.org/users/Belass/) | 0.899 | 0.809 | Create pixel-level vector representations of raster data by flattening VV, VH, and auxiliary bands. Train a set of CatBoostClassifiers over 1000 iterations using sequential, regional, k-fold, and stratified k-fold sampling, and average predictions. Finetune pretrained U-Net CNNs with EfficientNet-B0 and B4 backbones using samples of standardized, 3-channel images (VV, VH, & NASADEM) with flip, rotation, crop, and blurring augmentations. Train one model on weakly performing images. Ensemble CatBoost and U-Net predictions using the maximum output. Missing pixels are treated as zero.
2   | [Max_Lutz](https://www.drivendata.org/users/Max_Lutz/) | 0.908 | 0.807 | Create 9-channel images by stacking VH, VV, and auxiliary input bands. Clip VV and VH bands, scale all values to 0-255, and concatenate additional flipped and rotated augmentations. Train 3 randomly initialized U-Net CNNs using different train-test-splits and a consistent loss function of dice loss with a squared denominator. Ensemble predictions using the mean.
3   | [loweew](https://www.drivendata.org/users/loweew/) | 0.898 | 0.804 | Create standardized 9-channel images by stacking VH, VV, and auxiliary input bands. Apply augmentations including cropping, rotation, translation, scaling, blurring, affine transformation, grid and optical distortion, grid shuffle, and brightness contrast. Use a U-Net++ CNN with an EfficientNet-B8 backbone pretrained using AdvProp, which treats adversarial examples as additional input, to assign difficult floods to the training set. Finetune U-Net++ CNNs with an EfficientNet-L2 backbone pretrained using Noisy Student Training, using Focal Tversky loss to address data imbalance. Take a “cross-validation as a jury” approach to average predictions.

Additional solution details, including tips for using subsets of the winning models, can be found in the `reports` folder inside the directory for each submission. To request access to pre-computed survey weights, please [contact us](https://www.drivendata.org/contact/).

**Benchmark Blog Post: [How to Map Floodwater from Radar Imagery using Semantic Segmentation](https://www.drivendata.co/blog/detect-floodwater-benchmark/)**

## Competition Sponsor
<p align="center">
	<img src="https://drivendata-public-assets.s3.amazonaws.com/microsoft-logo-675x280.jpg" width="300"/>
</p>
