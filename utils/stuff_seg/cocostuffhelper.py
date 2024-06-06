__author__ = 'hcaesar'

# Helper functions used to convert between different formats for the
# COCO Stuff Segmentation Challenge.
#
# Note: Some functions use the Pillow image package, which may need
# to be installed manually.
#
# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Licensed under the Simplified BSD License [see coco/license.txt]

import numpy as np
from pycocotools import mask
from PIL import Image, ImagePalette # For indexed images
import matplotlib # For Matlab's color maps

def segmentationToCocoMask(labelMap, labelId):
    '''
    Encodes a segmentation mask using the Mask API.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param labelId: the label from labelMap that will be encoded
    :return: Rs - the encoded label mask for label 'labelId'
    '''
    labelMask = labelMap == labelId
    labelMask = np.expand_dims(labelMask, axis=2)
    labelMask = labelMask.astype('uint8')
    labelMask = np.asfortranarray(labelMask)
    Rs = mask.encode(labelMask)
    assert len(Rs) == 1
    Rs = Rs[0]

    return Rs

def segmentationToCocoResult(labelMap, imgId, stuffStartId=92):
    '''
    Convert a segmentation map to COCO stuff segmentation result format.
    :param labelMap: [h x w] segmentation map that indicates the label of each pixel
    :param imgId: the id of the COCO image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Get stuff labels
    shape = labelMap.shape
    if len(shape) != 2:
        raise Exception(('Error: Image has %d instead of 2 channels! Most likely you '
        'provided an RGB image instead of an indexed image (with or without color palette).') % len(shape))
    [h, w] = shape
    assert h > 0 and w > 0
    labelsAll = np.unique(labelMap)
    labelsStuff = [i for i in labelsAll if i >= stuffStartId]

    # Add stuff annotations
    anns = []
    for labelId in labelsStuff:

        # Create mask and encode it
        Rs = segmentationToCocoMask(labelMap, labelId)

        # Create annotation data and add it to the list
        anndata = {}
        anndata['image_id'] = int(imgId)
        anndata['category_id'] = int(labelId)
        anndata['segmentation'] = Rs
        anns.append(anndata)
    return anns

def cocoSegmentationToSegmentationMap(coco, imgId, checkUniquePixelLabel=True, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the id of the COCO image
    :param checkUniquePixelLabel: (optional) whether every pixel can have at most one label
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: labelMap - [h x w] segmentation map that indicates the label of each pixel
    '''

    # Init
    curImg = coco.imgs[imgId]
    imageSize = (curImg['height'], curImg['width'])
    labelMap = np.zeros(imageSize)

    # Get annotations of the current image (may be empty)
    imgAnnots = [a for a in coco.anns.values() if a['image_id'] == imgId]
    if includeCrowd:
        annIds = coco.getAnnIds(imgIds=imgId)
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
    imgAnnots = coco.loadAnns(annIds)

    # Combine all annotations of this image in labelMap
    #labelMasks = mask.decode([a['segmentation'] for a in imgAnnots])
    for a in range(0, len(imgAnnots)):
        labelMask = coco.annToMask(imgAnnots[a]) == 1
        #labelMask = labelMasks[:, :, a] == 1
        newLabel = imgAnnots[a]['category_id']

        if checkUniquePixelLabel and (labelMap[labelMask] != 0).any():
            raise Exception('Error: Some pixels have more than one label (image %d)!' % (imgId))

        labelMap[labelMask] = newLabel

    return labelMap

def pngToCocoResult(pngPath, imgId, stuffStartId=92):
    '''
    Reads an indexed .png file with a label map from disk and converts it to COCO result format.
    :param pngPath: the path of the .png file
    :param imgId: the COCO id of the image (last part of the file name)
    :param stuffStartId: (optional) index where stuff classes start
    :return: anns    - a list of dicts for each label in this image
       .image_id     - the id of the COCO image
       .category_id  - the id of the stuff class of this annotation
       .segmentation - the RLE encoded segmentation of this class
    '''

    # Read indexed .png file from disk
    im = Image.open(pngPath)
    labelMap = np.array(im)

    # Convert label map to COCO result format
    anns = segmentationToCocoResult(labelMap, imgId, stuffStartId)
    return anns

def cocoSegmentationToPng(coco, imgId, pngPath, includeCrowd=False):
    '''
    Convert COCO GT or results for a single image to a segmentation map and write it to disk.
    :param coco: an instance of the COCO API (ground-truth or result)
    :param imgId: the COCO id of the image (last part of the file name)
    :param pngPath: the path of the .png file
    :param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
    :return: None
    '''

    # Create label map
    labelMap = cocoSegmentationToSegmentationMap(coco, imgId, includeCrowd=includeCrowd)
    labelMap = labelMap.astype(np.int8)

    # Get color map and convert to PIL's format
    cmap = getCMap()
    cmap = (cmap * 255).astype(int)
    padding = np.zeros((256-cmap.shape[0], 3), np.int8)
    cmap = np.vstack((cmap, padding))
    cmap = cmap.reshape((-1))
    assert len(cmap) == 768, 'Error: Color map must have exactly 256*3 elements!'

    # Write to png file
    png = Image.fromarray(labelMap).convert('P')
    png.putpalette(cmap)
    png.save(pngPath, format='PNG')

def getCMap(stuffStartId=92, stuffEndId=182, cmapName='jet', addThings=True, addUnlabeled=True, addOther=True):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param stuffStartId: (optional) index where stuff classes start
    :param stuffEndId: (optional) index where stuff classes end
    :param cmapName: (optional) Matlab's name of the color map
    :param addThings: (optional) whether to add a color for the 91 thing classes
    :param addUnlabeled: (optional) whether to add a color for the 'unlabeled' class
    :param addOther: (optional) whether to add a color for the 'other' class
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = stuffEndId - stuffStartId + 1
    cmapGen = matplotlib.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = matplotlib.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = matplotlib.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]

    # Add black (or any other) color for each thing class
    if addThings:
        thingsPadding = np.zeros((stuffStartId - 1, 3))
        cmap = np.vstack((thingsPadding, cmap))

    # Add black color for 'unlabeled' class
    if addUnlabeled:
        cmap = np.vstack(((0.0, 0.0, 0.0), cmap))

    # Add yellow/orange color for 'other' class
    if addOther:
        cmap = np.vstack((cmap, (1.0, 0.843, 0.0)))

    return cmap
