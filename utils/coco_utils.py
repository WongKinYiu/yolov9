import cv2
import json
import multiprocessing
import numpy as np
import os
import PIL.Image as Image
import time

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as maskUtils
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

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

# details: https://cocodataset.org/#panoptic-eval
'''
tree-merged(184): branch(94), tree(169), bush(97), leaves(129)
fence-merged(185): cage(99), fence(113), railing(146)
ceiling-merged(186): ceiling-tile(103), ceiling-other(102)
sky-other-merged(187): clouds(106), sky-other(157), fog(120)
cabinet-merged(188): cupboard(108), cabinet(98)
table-merged(189): desk-stuff(110), table(165)
floor-other-merged(190): floor-marble(114), floor-other(115), floor-tile(117)
pavement-merged(191): floor-stone(116), pavement(140)
mountain-merged(192): hill(127), mountain(135)
grass-merged(193): moss(134), grass(124), straw(163)
dirt-merged(194): mud(136), dirt(111)
paper-merged(195): napkin(137), paper(139)
food-other-merged(196): salad(153), vegetable(170), food-other(121)
building-other-merged(197): skyscraper(158), building-other(96)
rock-merged(198): stone(162), rock(150)
wall-other-merged(199): wall-other(173), wall-concrete(172), wall-panel(174)
rug-merged(200): mat(131), rug(152), carpet(101)
'''
stuff_to_panoptic_mapping = {
    94: 184, 169: 184, 97: 184, 129: 184,   # 184
    99: 185, 113: 185, 146: 185,            # 185
    103: 186, 102: 186,                     # 186
    106: 187, 157: 187, 120: 187,           # 187
    108: 188, 98: 188,                      # 188
    110: 189, 165: 189,                     # 189
    114: 190, 115: 190, 117: 190,           # 190
    116: 191, 140: 191,                     # 191
    127: 192, 135: 192,                     # 192
    134: 193, 124: 193, 163: 193,           # 193
    136: 194, 111: 194,                     # 194
    137: 195, 139: 195,                     # 195
    153: 196, 170: 196, 121: 196,           # 196
    158: 197, 96: 197,                      # 197
    162: 198, 150: 198,                     # 198
    173: 199, 172: 199, 174: 199,           # 199
    131: 200, 152: 200, 101: 200,           # 200
}

def getCocoIds(name = 'semantic'):
    if 'instances' == name:
        return all_instances_ids
    elif 'stuff' == name:
        return 	all_stuff_ids
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

def idToPanopticId(id):
    if (id in all_stuff_ids) and (id not in panoptic_stuff_ids):
        # merged class or removed class
        if id in stuff_to_panoptic_mapping:
            # merged class
            return stuff_to_panoptic_mapping[id]
        # removed class should be setting to unlabeled
        return 0

    # reserved class or non-stuff class
    return id

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


''' Panoptic '''

class IdGenerator():
    '''
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    (X) The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    (O) The RGB encoding used is ID = R + 256 * G + 256 * 256 * B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    '''
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if 0 == category['isthing']:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(
                low = -max_dist,
                high = max_dist + 1,
                size = 3
            )
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if 0 == category['isthing']:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and (3 == len(color.shape)):
        if np.uint8 == color.dtype:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


'''
Modified cocodataset/panopticapi
This script converts panoptic COCO format to detection COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data. All segments will be stored in RLE format.

Additional option:
- using option '--things_only' the script can discard all stuff
segments, saving segments of things classes only.
'''

def convert_panoptic_to_detection_coco_format_single_core(
    proc_id, annotations_set, categories, segmentations_folder, things_only
):
    annotations_detection = []
    for working_idx, annotation in enumerate(annotations_set):
        if 0 == working_idx % 100:
            print(f'Core: {proc_id}, {working_idx} from {len(annotations_set)} images processed')

        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, file_name)), dtype = np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))
        pan = rgb2id(pan_format)

        for segm_info in annotation['segments_info']:
            if things_only and (1 != categories[segm_info['category_id']]['isthing']):
                continue
            mask = (pan == segm_info['id']).astype(np.uint8)
            mask = np.expand_dims(mask, axis = 2)
            segm_info.pop('id')
            segm_info['image_id'] = annotation['image_id']
            rle = maskUtils.encode(np.asfortranarray(mask))[0]
            rle['counts'] = rle['counts'].decode('utf8')
            segm_info['segmentation'] = rle
            annotations_detection.append(segm_info)

    print(f'Core: {proc_id}, all {len(annotations_set)} images processed')
    return annotations_detection


def convert_panoptic_to_detection_coco_format(
    input_json_file,
    segmentations_folder,
    output_json_file,
    categories_json_file,
    things_only = False,
):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("CONVERTING...")
    print("COCO panoptic format:")
    print(f"\tSegmentation folder: {segmentations_folder}")
    print(f"\tJSON file: {input_json_file}")
    print("TO")
    print("COCO detection format")
    print(f"\tJSON file: {output_json_file}")
    if things_only:
        print("Saving only segments of things classes.")
    print("\n")

    print(f'Reading annotation information from {input_json_file}')
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {
        category['id']: category for category in categories_list
    }

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print(f'Number of cores: {cpu_num}, images per core: {len(annotations_split[0])}')
    workers = multiprocessing.Pool(processes = cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(
            convert_panoptic_to_detection_coco_format_single_core,
            (proc_id, annotations_set, categories, segmentations_folder, things_only)
        )
        processes.append(p)
    annotations_coco_detection = []
    for p in processes:
        annotations_coco_detection.extend(p.get())
    for idx, ann in enumerate(annotations_coco_detection):
        ann['id'] = idx

    d_coco['annotations'] = annotations_coco_detection
    categories_coco_detection = []
    for category in d_coco['categories']:
        if things_only and (1 != category['isthing']):
            continue
        category.pop('isthing')
        category.pop('color')
        categories_coco_detection.append(category)
    d_coco['categories'] = categories_coco_detection
    with open(output_json_file, 'w') as f:
        json.dump(d_coco, f)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


'''
Modified cocodataset/panopticapi
This script converts detection COCO format to panoptic COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data.
'''

def convert_detection_to_panoptic_coco_format_single_core(
    proc_id, coco_detection, img_ids, categories, segmentations_folder
):
    id_generator = IdGenerator(categories)

    annotations_panoptic = []
    for working_idx, img_id in enumerate(img_ids):
        if 0 == working_idx % 100:
            print(f'Core: {proc_id}, {working_idx} from {len(img_ids)} images processed')
        img = coco_detection.loadImgs(int(img_id))[0]
        pan_format = np.zeros((img['height'], img['width'], 3), dtype = np.uint8)
        overlaps_map = np.zeros((img['height'], img['width']), dtype = np.uint32)

        anns_ids = coco_detection.getAnnIds(img_id)
        anns = coco_detection.loadAnns(anns_ids)

        panoptic_record = {}
        panoptic_record['image_id'] = img_id
        file_name = '{}.png'.format(img['file_name'].rsplit('.')[0])
        panoptic_record['file_name'] = file_name
        segments_info = []
        for ann in anns:
            if ann['category_id'] not in categories:
                raise Exception(
                    'Panoptic coco categories file does not contain \
                    category with id: {}'.format(ann['category_id'])
                )
            segment_id, color = id_generator.get_id_and_color(ann['category_id'])
            mask = coco_detection.annToMask(ann)
            overlaps_map += mask
            pan_format[mask == 1] = color
            ann.pop('segmentation')
            ann.pop('image_id')
            ann['id'] = segment_id
            segments_info.append(ann)

        if 0 != np.sum(1 < overlaps_map):
            raise Exception(f'Segments for image {img_id} overlap each other.')
        panoptic_record['segments_info'] = segments_info
        annotations_panoptic.append(panoptic_record)

        Image.fromarray(pan_format).save(os.path.join(segmentations_folder, file_name))

    print(f'Core: {proc_id}, all {len(img_ids)} images processed')
    return annotations_panoptic


def convert_detection_to_panoptic_coco_format(
    input_json_file,
    segmentations_folder,
    output_json_file,
    categories_json_file
):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = output_json_file.rsplit('.', 1)[0]
    if not os.path.isdir(segmentations_folder):
        print(f'Creating folder {segmentations_folder} for panoptic segmentation PNGs')
        os.mkdir(segmentations_folder)

    print("CONVERTING...")
    print("COCO detection format:")
    print(f"\tJSON file: {input_json_file}")
    print("TO")
    print("COCO panoptic format")
    print(f"\tSegmentation folder: {segmentations_folder}")
    print(f"\tJSON file: {output_json_file}")
    print('\n')

    coco_detection = COCO(input_json_file)
    img_ids = coco_detection.getImgIds()

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {
        category['id']: category for category in categories_list
    }

    cpu_num = multiprocessing.cpu_count()
    img_ids_split = np.array_split(img_ids, cpu_num)
    print(f'Number of cores: {cpu_num}, images per core: {len(img_ids_split[0])}')
    workers = multiprocessing.Pool(processes = cpu_num)
    processes = []
    for proc_id, img_ids in enumerate(img_ids_split):
        p = workers.apply_async(
            convert_detection_to_panoptic_coco_format_single_core,
            (proc_id, coco_detection, img_ids, categories, segmentations_folder)
        )
        processes.append(p)
    annotations_coco_panoptic = []
    for p in processes:
        annotations_coco_panoptic.extend(p.get())

    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    d_coco['annotations'] = annotations_coco_panoptic
    d_coco['categories'] = categories_list
    with open(output_json_file, 'w') as f:
        json.dump(d_coco, f)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))