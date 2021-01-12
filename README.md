# coco_dataset_resize
Python tool you can use to resize the images and bounding boxes of your COCO based dataset.

## Pre-requisites

- In order to use this tool you need to have python >=3.6 installed on your machine.

- Install the dependencies using the following command line :
```bash
pip install -r requirements.txt
```

## How to use this tool

A concrete example is worth more than a long speech : 

```bash
python coco_dataset_resize.py --images_dir="dataset/images/" --annotations_file="dataset/annotations.json" --image_width=512 --image_height=512 --output_ann_file="resized_dataset/annotations.json" --output_img_dir="resized_dataset/images/"
```
