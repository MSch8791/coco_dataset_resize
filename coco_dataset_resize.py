#################################################################################
#                                                                               #
#                               coco_dataset_resize                             #
#                 Author : Michaël Scherer (schererm8791@gmail.com)             #
#                                                                               #
#                              License : GPL v3.0                               #
#                                                                               #
#################################################################################

import argparse
import json
import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from collections import defaultdict


def resizeImageAndBoundingBoxes(imgFile, bboxes, inputW, inputH, targetImgW, targetImgH, outputImgFile):
    print("Reading image {0} ...".format(imgFile))
    img = cv2.imread(imgFile)

    if inputW > inputH:
        seq = iaa.Sequential([
                                iaa.Resize({"height": "keep-aspect-ratio", "width": targetImgW}),
                                iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
                            ])
    else:
        seq = iaa.Sequential([
                                iaa.Resize({"height": targetImgH, "width": "keep-aspect-ratio"}),
                                iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
                            ])
    
    image_aug, bbs_aug = seq(image=img, bounding_boxes=bboxes)

    print("Writing resized image {0} ...".format(outputImgFile))
    cv2.imwrite(outputImgFile, image_aug)
    print("Resized image {0} written successfully.".format(outputImgFile))

    return bbs_aug


if __name__ == "__main__":

    ia.seed(1)

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--images_dir", required=True, help="Directory where are located the images referenced in the annotations file")
    ap.add_argument("-a", "--annotations_file", required=True, help="COCO JSON format annotations file")
    ap.add_argument("-w", "--image_width", required=True, help="Target image width")
    ap.add_argument("-t", "--image_height", required=True, help="Target image height")
    ap.add_argument("-o", "--output_ann_file", required=True, help="Output annotations file")
    ap.add_argument("-f", "--output_img_dir", required=True, help="Output images directory")

    args = vars(ap.parse_args())

    imageDir                = args['images_dir']
    annotationsFile         = args['annotations_file']
    targetImgW              = int(args['image_width'])
    targetImgH              = int(args['image_height'])
    outputImageDir          = args['output_img_dir']
    outputAnnotationsFile   = args['output_ann_file']

    print("Loading annotations file...")
    data = json.load(open(annotationsFile, 'r'))
    print("Annotations file loaded.")

    print("Building dictionnaries...")
    anns    = defaultdict(list)
    annsIdx = dict()
    for i in range(0, len(data['annotations'])):
        anns[data['annotations'][i]['image_id']].append(data['annotations'][i])
        annsIdx[data['annotations'][i]['id']] = i
        data['annotations'][i]['category_id'] = 1
    print("Dictionnaries built.")

    for img in data['images']:
        print("Processing image file {0} and its bounding boxes...".format(img['file_name']))

        annList = anns[img['id']]

        bboxesList = []
        for ann in annList:
            bboxData = ann['bbox']
            bboxesList.append(BoundingBox(x1=bboxData[0], y1=bboxData[1], x2=bboxData[0] + bboxData[2], y2=bboxData[1] + bboxData[3]))

        imgFullPath         = os.path.join(imageDir, img['file_name'])
        outputImgFullPath   = os.path.join(outputImageDir, img['file_name'])
        outputDir           = os.path.dirname(outputImgFullPath)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        outNewBBoxes = resizeImageAndBoundingBoxes(imgFullPath, bboxesList, int(img['width']), int(img['height']), targetImgW, targetImgH, outputImgFullPath)

        for i in range(0, len(annList)):
            annId = annList[i]['id']
            data['annotations'][annsIdx[annId]]['bbox'][0] = outNewBBoxes[i].x1
            data['annotations'][annsIdx[annId]]['bbox'][1] = outNewBBoxes[i].y1
            data['annotations'][annsIdx[annId]]['bbox'][2] = outNewBBoxes[i].x2 - outNewBBoxes[i].x1
            data['annotations'][annsIdx[annId]]['bbox'][3] = outNewBBoxes[i].y2 - outNewBBoxes[i].y1
        
        img['width']    = targetImgW
        img['height']   = targetImgH
    
    print("Writing modified annotations to file...")
    with open(outputAnnotationsFile, 'w') as outfile:
        json.dump(data, outfile)
    
    print("Finished.")






