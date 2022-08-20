
Garbage detection - v1 2022-08-17 11:14pm
==============================

This dataset was exported via roboflow.com on August 18, 2022 at 2:18 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

It includes 175 images.
Garbage are annotated in YOLO v5 PyTorch format.

The following pre-processing was applied to each image:
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* 50% probability of vertical flip
* Random rotation of between -15 and +15 degrees
* Salt and pepper noise was applied to 5 percent of pixels


