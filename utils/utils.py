import numpy as np
import math
import json
import pandas as pd
from shapely import intersection, union
from shapely.geometry import Polygon
from sklearn.neighbors import BallTree
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as Polygon_mpl

####################################
# Functions for annotating dataset #
####################################
def format_image(img_id, filename, image_dims):
    return {'id': img_id,
            'file_name': filename,
            'width': image_dims[0],
            'height': image_dims[1]}

def format_annotation(ann_id, img_id, label_id, bbox):
    return {'id': ann_id,
            'image_id': img_id,
            'category_id': label_id,
            'bbox': bbox,
            'area': bbox[2] * bbox[3],
            'iscrowd': 0,
            'segmentation': []}

def get_single_ann_file(filepath_anns, splits):
    # Combine annotations from multiple splits into a single dict
    anns = {'categories': [], 'images': [], 'annotations': []}
    for split in splits:
        filepath_split = filepath_anns.split('.')
        filepath = '.'.join(filepath_split[:-1])
        ext = filepath_split[-1]
        with open(f'{filepath}_{split}.{ext}', 'r') as fp:
            anns_split = json.load(fp)
        for key, val in anns_split.items():
            anns[key].extend(val)
    anns['categories'] = anns_split['categories']
    return anns

#############################################
# Functions for street view panorama images #
#############################################

def haversine_distance(lat_1, lng_1, lat_2, lng_2):
    # Get haversine distance in meters between (lat1, lng1) and (lat2, lng2)
    earth_radius = 6371000
    lat_1, lng_1, lat_2, lng_2 = [float(x) for x in [lat_1, lng_1, lat_2, lng_2]]
    hlat = math.sin( math.radians((lat_2 - lat_1) / 2.0)) **2
    hlng = math.sin( math.radians((lng_2 - lng_1) / 2.0)) **2
    a = hlat + math.cos( math.radians(lat_1) ) * math.cos( math.radians(lat_2)) * hlng
    c = 2 * math.atan2( math.sqrt(a), math.sqrt(1 - a) )
    return earth_radius * c
    
def get_nearest_panos(lat, lng, min_dist, max_dist, panos):
    # Get all panos within within min_dist and max_dist of (lat, lng)
    nearest_panos = []
    for pid, pano in panos.items():
        pano_lat, pano_lng = float(pano['Location']['lat']), float(pano['Location']['lng'])
        dist = haversine_distance(lat, lng, pano_lat, pano_lng)
        if dist < min_dist:
            return []
        elif dist <= max_dist:
            nearest_panos.append((pano, dist))
    return sorted(nearest_panos, key=lambda x: x[1])

def get_pano_size(pano, streetview_zoom):
    # Downsample the image size
    # (the size defined in the pano file is if max zoom was used)
    max_zoom = int(pano['Location']['zoomLevels']) if 'zoomLevels' in pano['Location'].keys() else 5
    down = round(math.pow(2, max_zoom - streetview_zoom))
    pano_width = round(float(pano['Data']['image_width'])) // down
    pano_height = round(float(pano['Data']['image_height'])) // down

    return pano_width, pano_height

#####################################################################################
# Functions for transforming between geographical coordinates and pixel coordinates #
#####################################################################################

def geo_coords_to_streetview_pixel(pano, tree_lat, tree_lng):
    # Find the distance between camera and tree in all dimensions
    pano_lat = float(pano['Location']['original_lat']) if 'original_lat' in pano['Location'].keys() else pano['Location']['lat']
    pano_lng = float(pano['Location']['original_lng']) if 'original_lng' in pano['Location'].keys() else pano['Location']['lng']
    earth_radius = 6371000
    dx = math.cos( math.radians(pano_lat) ) * math.sin( math.radians(tree_lng - pano_lng) )
    dy = math.sin( math.radians(tree_lat - pano_lat))
    z = math.sqrt(dx * dx + dy * dy) * earth_radius * 1.2

    # Compute the angle of the tree both horizontally and vertically
    google_car_camera_height = 3.0 
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg']) * math.pi / 180
    tree_coords_height = 0
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw  
    while look_at_angle > 2 * math.pi: 
        look_at_angle = look_at_angle - 2 * math.pi
    while look_at_angle < 0: 
        look_at_angle = look_at_angle + 2 * math.pi
    tilt_angle = math.atan2(tree_coords_height - google_car_camera_height, z) - pitch
    
    # Return the pixel coordinates of the tree in the streetview panorama
    pano_width, pano_height = get_pano_size(pano, streetview_zoom=2)
    x = (pano_width * look_at_angle) / (2 * math.pi)
    y = pano_height // 2 - pano_height * tilt_angle / math.pi 
    
    return x, y 

def geo_coords_to_streetview_bbox(pano, tree_lat, tree_lng, tree_shape):
    # Find the distance between camera and tree in all dimensions
    pano_lat = pano['Location']['lat']
    pano_lng = pano['Location']['lng']
    earth_radius = 6371000
    dx = math.cos( math.radians(pano_lat) ) * math.sin( math.radians(tree_lng - pano_lng) )
    dy = math.sin( math.radians(tree_lat - pano_lat))
    z = math.sqrt(dx * dx + dy * dy) * earth_radius * 1.2

    # Compute the angle of the tree both horizontally and vertically
    pitch = 0
    yaw = float(pano['Projection']['pano_yaw_deg']) * math.pi / 180
    google_car_camera_height = 3.0 
    tree_coords_height = 0
    look_at_angle = math.pi + math.atan2(dx, dy) - yaw  
    while look_at_angle > 2 * math.pi: 
        look_at_angle = look_at_angle - 2 * math.pi
    while look_at_angle < 0: 
        look_at_angle = look_at_angle + 2 * math.pi
    tilt_angle = math.atan2(tree_coords_height - google_car_camera_height, z) - pitch

    # Return the pixel coordinates of the tree in the streetview panorama
    pano_width, pano_height = get_pano_size(pano, streetview_zoom=2)
    x = pano_width * look_at_angle / (2 * math.pi)
    y = pano_height // 2 - pano_height * tilt_angle / math.pi 
    
    # Compute the pixel coordinates of the bounding box around the tree
    x1 = pano_width * (math.atan2(- tree_shape[0] / 2, z) + look_at_angle) / (2 * math.pi)
    x2 = pano_width * (math.atan2(tree_shape[0] / 2, z) + look_at_angle) / (2 * math.pi)
    y1 = pano_height // 2 - pano_height * (math.atan2(tree_coords_height + tree_shape[1] - google_car_camera_height, z) - pitch) / math.pi
    y2 = y

    # Correct the coordinates to account for the distortion in the top of the image
    # The closer to the top of the image the top of the tree is,
    # the more the top of its bounding box should be shifted towards the top of the image
    f_y = 1.02
    m = pano_height / 2
    y1_new = m - (m - y1) ** f_y
    # .. and the more each side of its bounding box should be shifted away from the center of the tree
    f_x = 0.03
    x1_new = x - (x - x1) * ((m - y1) ** f_x)
    x2_new = x - (x - x2) * ((m - y1) ** f_x)
    
    return x1_new, max(0, y1_new), x2_new, y2, x, y

def streetview_pixel_to_geo_coords(pano, pano_pixel_x, pano_pixel_y):
    # Compute the angle of the tree with respect to the camera both horizontally and vertically
    pitch = 0
    pano_width, pano_height = get_pano_size(pano, streetview_zoom=2)
    look_at_angle = pano_pixel_x * (2 * math.pi) / pano_width
    tilt_angle = (pano_height // 2 - pano_pixel_y) * math.pi / pano_height + pitch
    
    # Find the distance between camera and tree in all dimensions
    earth_radius = 6371000
    google_car_camera_height = 3.0
    tree_coords_height = 0
    yaw = float(pano['Projection']['pano_yaw_deg']) * math.pi / 180
    z = ((tree_coords_height - google_car_camera_height) / math.tan( min(-1e-2, tilt_angle) )) / 1.2
    dx = math.sin(look_at_angle - math.pi + yaw) * z / earth_radius
    dy = math.cos(look_at_angle - math.pi + yaw) * z / earth_radius

    # Return the geolocation of the tree
    pano_lat = float(pano['Location']['original_lat']) if 'original_lat' in pano['Location'].keys() else pano['Location']['lat']
    pano_lng = float(pano['Location']['original_lng']) if 'original_lng' in pano['Location'].keys() else pano['Location']['lng']    
    tree_lat = pano_lat + math.degrees( math.asin(dy) )
    tree_lng = pano_lng + math.degrees( math.asin(dx / math.cos( math.radians(pano_lat) )) )

    return tree_lat, tree_lng

##########################################################
# Functions for computing overlap between bounding boxes #
##########################################################

def iou(bbox_1, bbox_2):
    # The area of the intersection divided by the area of union
    boxes = [[[box[0], box[1]], 
              [box[0]+box[2], box[1]], 
              [box[0]+box[2], box[1]+box[3]], 
              [box[0], box[1]+box[3]]] for box in [bbox_1, bbox_2]]
    polys = [Polygon(box) for box in boxes]
    iou = intersection(*polys).area / union(*polys).area
    return iou

def io2(bbox_1, bbox_2):
    # The intersection area divided by the area of the
    # bbox the furthest from the camera
    boxes = [[[box[0], box[1]], 
              [box[0]+box[2], box[1]], 
              [box[0]+box[2], box[1]+box[3]], 
              [box[0], box[1]+box[3]]] for box in [bbox_1, bbox_2]]
    polys = [Polygon(box) for box in boxes]
    bbox_behind_idx = np.argmin([box[-1][-1] for box in boxes])
    io2 = intersection(*polys).area / polys[bbox_behind_idx].area
    return io2

def get_overlap_score(ann_old, ann_new, img_width, x_diff_min, x_diff_max):
    # Get part of bboxes inside image
    ann_bboxes = []
    ann_xs = []
    for ann in [ann_old, ann_new]:
        ann_bbox = ann['bbox'].copy()
        ann_bbox[0] = max(0, ann_bbox[0])
        ann_bbox[2] = ann_bbox[2] if (ann_bbox[0] + ann_bbox[2]) < img_width else img_width - ann_bbox[0]
        ann_bboxes.append(ann_bbox)

        # Get pixel coordinates of trees at ground
        ann_x = ann['bbox'][0] + ann['bbox'][2] / 2
        ann_xs.append(ann_x)

    # Compute Intersection over Union
    overlap_iou = iou(*ann_bboxes)
    # Compute Intersection over the area of the new bbox (Io2)
    overlap_io2 = io2(*ann_bboxes)

    # Compute the overlap_x_diff if the bboxes are overlapping
    if (overlap_iou > 0) & (overlap_io2 > 0):
        # Compute the distance between the x coordinates of each tree
        # divided by the width of the image
        x_diff_frac = abs(ann_xs[0] - ann_xs[1]) / img_width
        # Convert values to make them high if the x distance is low
        overlap_x_diff = 1 - x_diff_frac
        # Normalize by the minimum- and maximum value of the x distances 
        # of all overlapping bounding boxes
        if x_diff_min != x_diff_max:
            overlap_x_diff = (overlap_x_diff - (1 - x_diff_max)) / ((1 - x_diff_min) - (1 - x_diff_max))
        # Make it zero if it's lower than the minimum x distance
        # (otherwise it would be negative)
        overlap_x_diff = max(0, overlap_x_diff)
    else:
        overlap_x_diff = 0

    # Combine the three overlap measures into a single overlap score
    # based on the IoU itself, a weighted average of the IoB and x distance, and the x distance itself
    overlap_all = max([overlap_iou, (overlap_io2 * 0.6 + overlap_x_diff * 1.4)/2, overlap_x_diff])
    return overlap_all

####################################
# Functions for making predictions #
####################################

def split_cluster(cluster, img_ids_cluster, ann_idxs):
    # If two or more predicted trees from the same image are in the same cluster,
    # split cluster and assign remaining trees to one of the new clusters
    cluster = np.array(cluster)

    # Find the image with the highest number of trees in the cluster
    img_id_most_common = max(img_ids_cluster, key=list(img_ids_cluster).count)

    # Use the trees in this image as new cluster centers
    tree_idxs = [i for i, img_id in enumerate(img_ids_cluster) if img_id == img_id_most_common]
    tree_locs = cluster[tree_idxs]
    clusters_new = [[loc] for loc in tree_locs]
    ann_idxs_new = [[ann_idx] for ann_idx in ann_idxs[tree_idxs]]
    
    # Assign all remaining trees to one of the new clusters
    for i, (loc, ann_idx) in enumerate(zip(cluster, ann_idxs)):
        if i not in tree_idxs:
            closest_cluster = np.argmin([haversine_distance(*loc, *loc_cluster) for loc_cluster in tree_locs])
            clusters_new[closest_cluster].append(loc)
            ann_idxs_new[closest_cluster].append(ann_idx)

    return clusters_new, ann_idxs_new

def get_weighted_cluster_center(cluster_locs, weights):
    center = [sum(cluster_locs[:, i] * weights) / sum(weights) for i in range(cluster_locs.shape[1])]
    return center

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise

def cluster_points(cluster_locs, ann_idxs, max_dist_cluster):
    # Cluster predicted trees within distance from each other
    kms_per_radian = 6371.0088
    epsilon = max_dist_cluster * 1e-3 / kms_per_radian
    clustering = AgglomerativeClustering(affinity=pairwise.PAIRWISE_DISTANCE_FUNCTIONS['haversine'],#'haversine',
                                         distance_threshold=epsilon,
                                         linkage='average',
                                         n_clusters=None).fit(np.radians(cluster_locs))
    anns_cluster_labels = clustering.labels_
    clusters_locs = [np.array(cluster_locs[anns_cluster_labels == cluster_label]) for cluster_label in range(len(set(anns_cluster_labels)))]
    ann_idxs_clusters = [np.array(ann_idxs[anns_cluster_labels == cluster_label]) for cluster_label in range(len(set(anns_cluster_labels)))]
    
    return clusters_locs, anns_cluster_labels, ann_idxs_clusters

def predict_locations(max_dist_cluster, max_dist_camera, anns_pred, panos, ann_to_cluster, last_cluster_label, filepath_preds=False):
    # Combine predicted annotations in images into final predicted geographical locations

    # Get image of each predicted annotated trees
    img_ids = [ann['image_id'] for ann in anns_pred['annotations']]
    imgs = [[img for img in anns_pred['images'] if img['id'] == img_id][0] for img_id in img_ids]
    pano_ids = [img['file_name'].split('.')[0].rstrip('_z2') for img in imgs]
    panos_imgs = [panos[pano_id] for pano_id in pano_ids]

    ann_idxs = range(len(anns_pred['annotations']))
    cluster_labels = np.full_like(ann_idxs, -2)
    locs, tree_widths, dists_to_camera = [], [], []
    for ann_idx in ann_idxs:

        # Get geolocations of predicted annotated trees
        bbox = anns_pred['annotations'][ann_idx]['bbox']
        x1, x2, y1, y2 = bbox[0], bbox[0]+bbox[2], bbox[1], bbox[1]+bbox[3]
        loc_pixels = (x1 + (x2-x1)/2, y2)
        loc_coords = streetview_pixel_to_geo_coords(panos_imgs[ann_idx], *loc_pixels)
        locs.append(loc_coords)

        # Get distance from prediction to camera
        pano_ann = panos_imgs[ann_idx]
        dist_to_camera = haversine_distance(*loc_coords, pano_ann['Location']['lat'], pano_ann['Location']['lng'])
        dists_to_camera.append(dist_to_camera)
        if dist_to_camera > max_dist_camera+5:
            cluster_labels[ann_idx] = -1

        # Get tree width of predicted annotated trees
        loc_coords_left = streetview_pixel_to_geo_coords(pano_ann, x1, y2)
        loc_coords_right = streetview_pixel_to_geo_coords(pano_ann, x2, y2)
        tree_width = haversine_distance(*loc_coords_left, *loc_coords_right)
        tree_widths.append(tree_width)

    locs, tree_widths, dists_to_camera = np.array(locs), np.array(tree_widths), np.array(dists_to_camera)
    ann_idxs = [ann_idx for ann_idx in ann_idxs if cluster_labels[ann_idx] != -1]


    # Get scores of predicted annotated trees
    scores = np.array([ann['score'] for ann in anns_pred['annotations']])
    
    # Cluster predictions above score threshold, 
    # add all predictions within max_dist_cluster of a cluster center independently of the confidence score,
    # repeat step for remaining trees without a cluster
    for score_th in [.8, 0.]:

        # Get predictions with confidence above threshold
        anns_idxs_round = np.array([ann_idx for ann_idx in ann_idxs if scores[ann_idx] >= score_th])

        # Cluster predictions
        clusters_locs, cluster_labels_round, ann_idxs_round_clusters = cluster_points(locs[anns_idxs_round], anns_idxs_round, max_dist_cluster)
        cluster_labels[anns_idxs_round] = cluster_labels_round + last_cluster_label

        # Get cluster centers
        weights = [(1 / np.array(dists_to_camera[ann_idxs_round_cluster])) * scores[ann_idxs_round_cluster] for ann_idxs_round_cluster in ann_idxs_round_clusters]
        cluster_centers = [get_weighted_cluster_center(np.array(cluster_locs), weight) for cluster_locs, weight in zip(clusters_locs, weights)]

        # Assign predictions outside of clusters to closest cluster within max_dist
        ball_tree = BallTree(np.radians(cluster_centers),
                            metric = 'haversine')
        for ann_idx in ann_idxs:
            if cluster_labels[ann_idx] == -2:
                dist, cluster_idx = get_closest_tree(ball_tree, locs[ann_idx])
                if dist <= max_dist_cluster:
                    cluster_labels[ann_idx] = cluster_idx + last_cluster_label

        # Remove trees belonging to clusters
        ann_idxs = [ann_idx for ann_idx in ann_idxs if cluster_labels[ann_idx] == -2]
        
        last_cluster_label += len(cluster_centers)

    # Get information about clusters
    clusters_sizes, clusters_centers, clusters_scores, clusters_tree_widths = [], [], [], []
    cluster_labels_sorted = sorted(list(set(cluster_labels)))
    for cluster_label in cluster_labels_sorted:

        # Get predictions in cluster
        ann_idxs_cluster = [ann_idx for ann_idx, c_label in enumerate(cluster_labels) if c_label == cluster_label]
        anns_cluster = [anns_pred['annotations'][ann_idx] for ann_idx in ann_idxs_cluster]
        for ann in anns_cluster:
            ann_to_cluster[ann['id']] = int(cluster_label)
        if cluster_label == -1:
            continue
        clusters_sizes.append(len(anns_cluster))
        
        # Use max confidence score of predictions in cluster as final cluster score
        cluster_scores = scores[ann_idxs_cluster]
        clusters_scores.append(max(cluster_scores))

        # Get distances from trees to camera
        cluster_dists_to_camera = dists_to_camera[ann_idxs_cluster]

        # Use the inverse distances to camera multiplied by the confidence scores as weight
        weights = (1 / np.array(cluster_dists_to_camera)) * cluster_scores

        # Get cluster center
        cluster_locs = locs[ann_idxs_cluster]
        cluster_center = get_weighted_cluster_center(cluster_locs, weights)
        clusters_centers.append(cluster_center)

        # Compute mean tree width of predictions in cluster
        cluster_tree_widths = tree_widths[ann_idxs_cluster]
        cluster_tree_width = sum(cluster_tree_widths * weights) / sum(weights)
        clusters_tree_widths.append(cluster_tree_width)

    # Create df with information about clusters
    clusters_df = pd.DataFrame({'id': [cluster_label for cluster_label in cluster_labels_sorted if cluster_label != -1],
                                'lat': [loc[0] for loc in clusters_centers],
                                'lng': [loc[1] for loc in clusters_centers],
                                'n_annotated_trees': clusters_sizes,
                                'max_score': clusters_scores,
                                'mean_tree_width': clusters_tree_widths})
    if filepath_preds:
        clusters_df.to_csv(filepath_preds, index=False)

    # print(f"\tMean n. of predicted trees per cluster: {len(anns_pred['annotations']) / n_clusters:.2f}")

    return clusters_df, ann_to_cluster, last_cluster_label

############################
# Functions for evaluation #
############################

def get_closest_tree(ball_tree, loc):
    # Get closest tree in ball_tree
    result = ball_tree.query([np.radians(loc)])
    earth_radius = 6371000
    return result[0][0][0] * earth_radius, result[1][0][0]

def match_preds_to_gts(trees_pred, trees_gt, max_dist_tp):
    # Create ball tree from coordinates of gt trees
    ball_tree = BallTree(np.radians(trees_gt[['lat', 'lng']].values),
                         metric = 'haversine')

    # Get closest gt tree for each predicted tree
    closest_trees = []
    for _, tree_pred in trees_pred.iterrows():
        dist, tree_gt_idx = get_closest_tree(ball_tree, tree_pred[['lat', 'lng']].values)
        if dist <= max_dist_tp:
            tree_gt_id = int(trees_gt.iloc[tree_gt_idx]['id'])
            tree_pred_id = int(tree_pred['id'])
            closest_trees.append([tree_pred_id, tree_gt_id, dist])

    # Make sure every predicted tree is matched to no more than one gt tree and vice versa
    closest_trees.sort(key=lambda x: x[-1])
    pred_to_gt = {}
    while len(closest_trees) > 0:
        # Get gt- and predicted trees pair with the current shortest distance
        tree_pred_id, tree_gt_id, dist = closest_trees.pop(0)
        pred_to_gt[tree_pred_id] = tree_gt_id

        # Remove all other distances related to the matching predicted- and gt trees
        closest_trees = [elem for elem in closest_trees if  elem[1] != tree_gt_id]
    
    return pred_to_gt

# Get precisions and recalls at different score thresholds
def get_precision_recall(trees_pred, trees_gt, n_thresholds, max_dist_tp):
    precisions, recalls = [], []
    n_trees_gt = len(trees_gt)
    last_n_preds = None

    # Iterate score thresholds
    thresholds = np.linspace(0, 1, n_thresholds+1)
    for i, score_th in enumerate(thresholds):

        # Add (1, 0) if last score_th
        if i == n_thresholds:
            precisions.append(1.), recalls.append(0.)
            continue

        # Remove predictions below score_th
        trees_pred_th = trees_pred[trees_pred['max_score'] >= score_th]
        new_n_preds = len(trees_pred_th)
        if (i > 1) & (new_n_preds == last_n_preds):
            # Use computations from last score_th
            precision = precisions[-1]
            recall = recalls[-1]
        else:
            # Match predicted trees to gt trees
            pred_to_gt = match_preds_to_gts(trees_pred_th, trees_gt, max_dist_tp)
            n_matches = len(pred_to_gt)
            precision = n_matches / len(trees_pred_th)
            recall = 1 if i == 0 else n_matches / n_trees_gt
        precisions.append(precision), recalls.append(recall)
        last_n_preds = new_n_preds

    return np.array(precisions), np.array(recalls), np.array(thresholds)

def show_anns(bboxes, ax):
    # Show bboxes as polygons that can be layered on top of an image
    ax.set_autoscale_on(False)
    polygons = []
    for bbox in bboxes:
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        polygon = [[bbox_x, bbox_y], 
                   [bbox_x, bbox_y + bbox_h], 
                   [bbox_x + bbox_w, bbox_y + bbox_h], 
                   [bbox_x + bbox_w, bbox_y]]
        polygons.append(Polygon_mpl(np.array(polygon)))
    edges = PatchCollection(polygons, facecolor='none', edgecolors='#ff7f0e', linewidths=1.5)
    ax.add_collection(edges)
