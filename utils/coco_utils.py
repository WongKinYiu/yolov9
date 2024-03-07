import cv2

from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# coco id: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
all_instances_ids = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
    61, 62, 63, 64, 65, 67, 70,
    72, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 84, 85, 86, 87, 88, 89, 90,
]

all_stuff_ids = [
    92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130,
    131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
    141, 142, 143, 144, 145, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170,
    171, 172, 173, 174, 175, 176, 177, 178, 179, 180,
    181, 182,
    # other
    183,
    # unlabeled
    0,
]

# panoptic id: https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
panoptic_stuff_ids = [
    92, 93, 95, 100,
    107, 109,
    112, 118, 119,
    122, 125, 128, 130,
    133, 138,
    141, 144, 145, 147, 148, 149,
    151, 154, 155, 156, 159,
    161, 166, 168,
    171, 175, 176, 177, 178, 180,
    181, 184, 185, 186, 187, 188, 189, 190,
    191, 192, 193, 194, 195, 196, 197, 198, 199, 200,
    # unlabeled
    0,
]

def getCocoIds(name = 'semantic'):
    if 'instances' == name:
        return all_instances_ids
    elif 'stuff' == name:
        return all_stuff_ids
    elif 'panoptic' == name:
        return all_instances_ids + panoptic_stuff_ids
    else: # semantic
        return all_instances_ids + all_stuff_ids

def getMappingId(index, name = 'semantic'):
    ids = getCocoIds(name = name)
    return ids[index]

def getMappingIndex(id, name = 'semantic'):
    ids = getCocoIds(name = name)
    return ids.index(id)

# convert ann to rle encoded string
def annToRLE(ann, img_size):
    h, w = img_size
    segm = ann['segmentation']
    if list == type(segm):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif list == type(segm['counts']):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle

# decode ann to mask martix
def annToMask(ann, img_size):
    rle = annToRLE(ann, img_size)
    m = maskUtils.decode(rle)
    return m

# convert mask to polygans
def convert_to_polys(mask):
    # opencv 3.2
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # before opencv 3.2
    # contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if 4 < len(contour):
            segmentation.append(contour)

    return segmentation
