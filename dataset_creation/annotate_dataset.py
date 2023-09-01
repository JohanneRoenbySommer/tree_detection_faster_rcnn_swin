import os
os.environ['USE_PYGEOS'] = '0'
import pandas as pd
import geopandas as gpd
from shapely import wkt

import numpy as np
import json
import math
import random
from collections import defaultdict, Counter
from random import shuffle

import sys
sys.path.insert(1, os.path.join(sys.path[0], '/Users/johannesommer/Documents/Universitetet/Kandidat/4. semester/Thesis/Code/'))
from utils.utils import format_annotation, format_image, get_nearest_panos, get_pano_size, geo_coords_to_streetview_bbox, io2, get_overlap_score

# Load panorama images
zoom = 2
with open(f'../data/raw/panos_{zoom}.json', 'r') as fp:
    panos = json.load(fp)

# Load tree inventory
trees = pd.read_csv(f'../data/raw/tree_inventory_cleaned.csv')
trees['geometry'] = trees['geometry'].apply(wkt.loads)
trees = gpd.GeoDataFrame(trees, geometry='geometry', crs='EPSG:4326')

# Find k most common labels
labels = trees['slaegt'].values
k = len(set(labels))
labels_top_k = [x[0] for x in Counter(labels).most_common(k)]

# Map trees, annotations and images
filenames = os.listdir('../data/streetviews')
mappings = {'pano_to_id': {filename: i for i, filename in enumerate(filenames)},
            'label_to_id': {label: i for i, label in enumerate(labels_top_k)},
            'ann_to_tree': {}}

# Prepare category part of annotation file
labels = [{'id': i, 'name': label} for label, i in mappings['label_to_id'].items()]

# Shuffle trees
trees = trees.sample(len(trees))

# Prepare saving information on images and annotations
anns_all = []
img_ids_added, ann_ids_added = set(), set()
ann_id = 0


# Iterate splits
splits = ['train', 'val', 'test']
for split in splits:
    img_id_to_img = {}
    img_to_anns = defaultdict(list)

    # Iterate trees in split
    for _, tree in trees[trees['split'] == split].iterrows():

        # Get geographical location of tree
        lat, lng = [float(tree[coor]) for coor in ['lat', 'lng']]

        # Get panorama images containing tree
        min_dist_tree_pano = 1.2
        max_dist_tree_pano = 18
        nearest_panos = get_nearest_panos(lat, lng, min_dist_tree_pano, max_dist_tree_pano, panos)

        # Remove panoramas captured before the tree was planted
        nearest_panos = [pano for pano in nearest_panos if pano[0]['Data']['image_date'][0] > tree['planteaar']]
        
        # Get ID of label
        label = tree['slaegt'] #if dataset_type == 'Cph' else tree['extended']['Common_Name']
        label_id = mappings['label_to_id'][label]

        # Add annotation of bounding box around tree on panorama image
        for (pano, dist_pano_tree) in nearest_panos:

            # if dataset_type == 'Cph':

            # Reduce tree height according to age of tree
            age = pano['Data']['image_date'][0] - tree['planteaar']
            min_height = 9
            max_height =  max(min_height, tree['max_height'])
            age_max_height = 15
            if max_height < 25:
                age_max_height = 10
            if max_height > 30:
                age_max_height = 25
            max_height_new = max_height * 0.6
            tree_height = max_height_new if age >= age_max_height else min_height + (age / age_max_height) * (max_height_new - min_height)
            
            # Reduce tree height further for pruned tree
            if (tree['ny_dm_element'] == 'Formede træer') & (max_height > 20):
                tree_height = max(4, 0.5 * tree_height)
            
            # Get size of bounding box based on tree height
            tree_sz = [0.6 * tree_height, tree_height]

            # Get pixel coordinates of bounding box
            x1, y1, x2, y2, x, y = geo_coords_to_streetview_bbox(pano, lat, lng, tree_sz)

            # Don't add trees that are placed in the middle of the road
            # (i.e. in the 10% of the left or right side of the image or in the 10% to the left and right of the middle of the image
            # when the tree is less than 2.5 meters from the camera)
            pano_width, pano_height = get_pano_size(pano, zoom)
            if (dist_pano_tree < 2.5) & ((0 <= x <= 0.1 * pano_width) or (0.4 * pano_width <= x <= 0.6 * pano_width) or (0.9 * pano_width <= x <= 1 * pano_width)):
                continue
                
            # Add panorama
            filename = pano['Location']['panoId'] + '.jpg' #+ '_z2.jpg'
            img_id = mappings['pano_to_id'][filename]
            if img_id not in img_id_to_img:
                img_formatted = format_image(img_id, filename, [pano_width, pano_height])
                img_id_to_img[img_id] = img_formatted
                img_ids_added.add(img_id)

            # Get annotation of bbox around tree on streetview panorama
            bbox = [x1, y1, x2 - x1, y2 - y1] # (left, upper, width_bbox, height_bbox)

            # Add annotation
            ann_formatted = format_annotation(ann_id, img_id, label_id, bbox)
            img_to_anns[img_id].append(ann_formatted)
            mappings['ann_to_tree'][ann_id] = tree['id']

            ann_id += 1


    # Prepare saving annotations and images
    anns_split = {'categories': labels, 
                  'images': list(img_id_to_img.values()), 
                  'annotations': []}
    
    
    # Find min and max Io2 (intersection area divided by the area of the bbox the furthest from the camera)
    # for all overlapping bboxes in the split
    # (will be used for normalization)
    x_diff_min = 1
    x_diff_max = 0
    for img_id, anns_img in img_to_anns.items():
        img = img_id_to_img[img_id]
        img_width = img['width']
        for i, ann1 in enumerate(anns_img):
            for ann2 in anns_img[i+1:]:
                overlap_io2 = io2(ann1['bbox'], ann2['bbox'])
                if overlap_io2 > 0:
                    ann1_x = ann1['bbox'][0] + ann1['bbox'][2] / 2
                    ann2_x = ann2['bbox'][0] + ann2['bbox'][2] / 2
                    x_diff = abs(ann1_x - ann2_x) / img_width
                    if x_diff < x_diff_min:
                        x_diff_min = x_diff
                    if x_diff > x_diff_max:
                        x_diff_max = x_diff

    # Remove bounding boxes covered by other bounding boxes in the same image
    for img_id, anns_img in img_to_anns.items():
        img = img_id_to_img[img_id]

        # Get annotations in image sorted by how far they are from the camera with the closest first
        anns_img_y = [ann['bbox'][1] + ann['bbox'][3] for ann in anns_img]
        anns_img_order = np.argsort(anns_img_y)[::-1]
        anns_img = [anns_img[i] for i in anns_img_order]

        # Add the first annotation to the list of annotations to be saved
        anns_img_new = [anns_img[0]]

        # Start at the second annotation
        ann_new_i = 1

        while ann_new_i < len(anns_img):
            ann_new_overlap = False

            # Get annotation
            ann_new = anns_img[ann_new_i]
            
            # Iterate all annotations that have already been added
            # (they are all closer to the camera than the current annotation)
            for ann_old in anns_img_new:

                # Check if the overlap score is above threshold
                overlap_th = 0.85
                overlap_all = get_overlap_score(ann_old, ann_new, img['width'], x_diff_min, x_diff_max)
                if (overlap_all >= overlap_th):
                    ann_new_overlap = True
                    break

            # TODO: Check that current annotation is not completely covered 
            # by already added annotations

            # Add current annotation 
            # if it doesn't overlap with any of the already added annotations
            if not ann_new_overlap:
                anns_img_new.append(ann_new)
                    
            ann_new_i += 1

        ann_ids_added.update([ann['id'] for ann in anns_img_new])
        anns_split['annotations'].extend(anns_img_new)

    # Shuffle images and annotations
    shuffle(anns_split['images'])
    shuffle(anns_split['annotations'])

    # Save annotations
    with open(f'../data/annotations/annotations_{split}.json', 'w') as fp:
        json.dump(anns_split, fp)

    # Save version of annotations with only a single label ('tree')
    anns_split['categories'] = [{'id': 0, 'name': 'tree'}]
    for ann in anns_split['annotations']:
        ann['category_id'] = 0
    with open(f'../data/annotations/annotations_{split}_single_label.json', 'w') as fp:
        json.dump(anns_split, fp)

    # Print statistics
    print(f'{split.capitalize()}:')
    print(f'\t{len(anns_split["images"])} images')
    print(f'\t{len(anns_split["annotations"])} annotations')

# Remove images and trees not used from mappings
mappings['pano_to_id'] = {filename: img_id for filename, img_id in mappings['pano_to_id'].items() if img_id in img_ids_added}
mappings['ann_to_tree'] = {ann_id: tree_id for ann_id, tree_id in mappings['ann_to_tree'].items() if ann_id in ann_ids_added}

# Save mappings
with open(f'../data/mappings/mappings.json', 'w') as fp:
    json.dump(mappings, fp)
