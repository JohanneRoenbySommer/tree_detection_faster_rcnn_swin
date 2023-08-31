# -*- coding: utf-8 -*-

# Code from https://github.com/cplusx/google-street-view-panorama-download/blob/master/streetview.py

"""
Original code is from https://github.com/robolyst/streetview
Functions added in this file are
download_panorama_v1, download_panorama_v2, download_panorama_v3
"""

import os
import pandas as pd
import json
import itertools
import time
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO

import sys
sys.path.insert(1, f'{sys.path[0]}/..')
from utils import haversine_distance

def get_panoids(lat, lng):
    # Get all close panos
    url = f'https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lng}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5'
    resp = requests.get(url, proxies=None)

    # Transform API response to json format
    resp_text_clean = resp.text.split('(')[1].split(')')[0].strip(' ')
    try:
        resp_json = json.loads(resp_text_clean)
    except json.decoder.JSONDecodeError:
        return []

    if len(resp_json) == 1:
        return []
    
    pano_infos = resp_json[1][5][0][3][0]

    # Remove panos captured below 2.5 m above sea level
    pano_infos = [pano_info for pano_info in pano_infos if len(pano_info) > 2]
    pano_infos = [pano_info for pano_info in pano_infos if pano_info[2][1]]
    pano_infos = [pano_info for pano_info in pano_infos if pano_info[2][1][0] > 2.5]

    # Remove panos captured at trails or in buildings
    pano_infos = [pano_info for pano_info in pano_infos if len(pano_info[2]) < 4]
    
    # Get the pano_id of the pano closest to lat, lng
    pano_ids = [pano_info[0][1] for pano_info in pano_infos]
    pano_dists = [haversine_distance(*pano_info[2][0][2:], lat, lng) for pano_info in pano_infos]
    panos_ids_sorted = [pano_info for _, pano_info in sorted(zip(pano_dists, pano_ids))]

    return panos_ids_sorted

def get_pano_info(pano_id, pano_id_recent=None):
    # Code from https://github.com/gladcolor/StreetView

    pano_info_dict = None
    pano_infos = []
    while not pano_info_dict:

        # Get response from Google Maps
        url = f'https://www.google.com/maps/photometa/v1?authuser=0&hl=zh-CN&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1szh-CN!2sus!3m3!1m2!1e2!2s{pano_id}!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3'
        resp = requests.get(url, proxies=None)
        resp_text_clean = resp.text.replace(")]}'\n", "")
        try:
            resp_json = json.loads(resp_text_clean)
        except json.decoder.JSONDecodeError:
            if len(pano_infos) > 1:
                pano_id = pano_infos[1][0][1]
            else:
                break
        if pano_id == pano_id_recent:
            return {'Location': {'panoId': pano_id}}
        pano_infos_all = resp_json[1][0][5][0][3][0]

        # Get date of capture and pano_info for first pano
        dates = [resp_json[1][0][6][7]] 
        pano_infos = [pano_infos_all.pop(0)]

        # Check if the first pano is captured in the summer
        use_first_pano = True
        if not (5 < dates[0][1] < 10):
            use_first_pano = False

        # Get date of capture and pano_info for the remaining panos if they exist
        if len(resp_json[1][0][5][0]) >= 9: #if there's more than one pano with a date from that location
            if resp_json[1][0][5][0][8]:
                dates += [x[1] for x in resp_json[1][0][5][0][8]]
                dates_idxs = [x[0] - 1 for x in resp_json[1][0][5][0][8]]
                pano_infos += [pano_infos_all[i] for i in dates_idxs]

                # Remove panos that are captured at trails or in buildings
                dates = [date for pano_info, date in zip(pano_infos, dates) if len(pano_info[2]) < 4]
                pano_infos = [pano_info for pano_info in pano_infos if len(pano_info[2]) < 4]

                # Remove panos captured below 3 meters above sea level
                dates = [date for pano_info, date in zip(pano_infos, dates) if pano_info[2][1]]
                dates = [date for pano_info, date in zip(pano_infos, dates) if pano_info[2][1][0] > 2.5]
                pano_infos = [pano_info for pano_info in pano_infos if pano_info[2][1]]
                pano_infos = [pano_info for pano_info in pano_infos if pano_info[2][1][0] > 2.5]

                # Remove panos with dates not in the summer
                pano_infos = [pano_infos[0]] + [pano_info for pano_info, date in zip(pano_infos[1:], dates[1:]) if (5 < date[1] < 10)]
                dates = [dates[0]] + [date for date in dates[1:] if (5 < date[1] < 10)]

                # Check if the first pano is the newest captured in the summer from that location
                if use_first_pano:
                    if len(dates) > 1:
                        dates_dt = [datetime(year=dates[i][0], month=dates[i][1], day=1) for i in range(2)]
                        if dates_dt[0] < dates_dt[1]:
                            use_first_pano = False

        # Save pano_info in a dict
        pano_infos.pop(0)
        if use_first_pano:
            pano_info_dict = {'Data': {'image_width': resp_json[1][0][2][2][1],
                                    'image_height': resp_json[1][0][2][2][0],
                                    'image_date': resp_json[1][0][6][7]},
                            'Projection': {'pano_yaw_deg': resp_json[1][0][5][0][1][2][0]},
                            'Location': {'panoId': resp_json[1][0][1][1],
                                        'lat': resp_json[1][0][5][0][1][0][2],
                                        'lng': resp_json[1][0][5][0][1][0][3]}}

        # Continue to next pano if the first pano is not suitable
        elif len(pano_infos) > 0:
            pano_id = pano_infos[0][0][1]
        else:
            break

    return pano_info_dict


def get_tiles_info(panoid, zoom=5):
    """
    Generate a list of a panorama's tiles and their position.

    The format is (x, y, filename, fileurl)
    """
    image_url = "https://cbk0.google.com/cbk?output=tile&panoid={}&zoom={}&x={}&y={}"
    coord = list(itertools.product(range(32), range(16)))
    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, zoom, x, y)) for x, y in coord]

    return tiles

def download_panorama(panoid, img_size, zoom=5):
    '''
    v3: save image information in a buffer. (v2: save image to dist then read)
    input:
        panoid: which is an id of image on google maps
        zoom: larger number -> higher resolution, from 1 to 5, better less than 3, some location will fail when zoom larger than 3
        disp: verbose of downloading progress, basically you don't need it
    output:
        panorama image (uncropped)
    '''

    # Get information on tiles needed to create the panorama image
    tiles_info = get_tiles_info(panoid, zoom=zoom)

    # Download tiles
    tile_width, tile_height = 512, 512
    img_w, img_h = img_size
    valid_tiles = []
    for i, tile in enumerate(tiles_info):
        x, y, fname, url = tile
        # Download tile if it is valid
        if (x * tile_width < img_w) and (y * tile_height < img_h):
            while True:
                try:
                    response = requests.get(url, stream=True)
                    break
                except requests.ConnectionError:
                    print("Connection error. Trying again in 2 seconds.")
                    time.sleep(2)
            # Check if error in response
            try:
                response.raise_for_status()
                tile_img = Image.open(BytesIO(response.content))
                valid_tiles.append(tile_img)
            except requests.exceptions.HTTPError:
                pass
            del response
    
    # Stich tiles into single panorama image
    if (len(valid_tiles) > 0) & (len(tiles_info) != (len(valid_tiles)+1)):
        panorama = Image.new('RGB', (img_w, img_h))
        i = 0
        for x, y, fname, url in tiles_info:
            if x*tile_width < img_w and y*tile_height < img_h: # tile is valid
                tile = valid_tiles[i]
                i+=1
                panorama.paste(im=tile, box=(x*tile_width, y*tile_height))
        return panorama
    else:
        return None


# Get coordinates of all trees
data_path = '../data'
fp = f'{data_path}/raw/tree_inventory_cleaned.csv'
trees = pd.read_csv(fp, usecols=['id', 'geometry', 'lat', 'lng'])
print('N. of trees:', len(trees))

# Get panos whose information has already been added to file
zoom = 2
with open(f'{data_path}/raw/panos_{zoom}.json', 'r') as fp:
    panos = json.load(fp)
pano_ids_added = panos.keys()

# Get panos that have already been downloaded
streetview_path = f'{data_path}/images/streetview_{zoom}'
pano_ids_loaded = set(file_name.split('.')[0] for file_name in os.listdir(streetview_path))

# Iterate trees
for idx, data in trees.iloc[14824:].iterrows():
    print(idx)
    pano_id_added = None
    
    # Get pano_ids for nearest panos
    lat, lng = float(data['lat']), float(data['lng'])
    pano_ids = get_panoids(lat, lng)
    
    # Iterate panos from closest to furthest
    n_panos_added = 0
    while pano_ids:
        pano_added, pano_loaded = False, False
        pano_id = pano_ids[0]

        # Get pano_info for newest tree from that location
        # (might not return same pano_id)
        pano_info = get_pano_info(pano_id, pano_id_added)
        if pano_info:
            pano_id = pano_info['Location']['panoId']
            
            # Check if pano_id has already been added for image
            if pano_id != pano_id_added:

                # Keep track of added panos
                pano_added = True if pano_id in pano_ids_added else False
                pano_loaded = True if pano_id in pano_ids_loaded else False

                # Save pano_info
                if not pano_added:
                    with open(f'{data_path}/raw/panos_{zoom}.json', 'r') as file_in:
                        pano_file = json.load(file_in)
                    pano_file[pano_id] = pano_info
                    with open(f'{data_path}/raw/panos_{zoom}.json', 'w') as file_out:
                        json.dump(pano_file, file_out)
                    pano_added = True

                if not pano_loaded:
                    # Get size of pano
                    max_zoom = 5
                    pano_size_orig = pano_info['Data']['image_width'], pano_info['Data']['image_height']
                    down = 2**(max_zoom-zoom)
                    pano_size = [x//down for x in pano_size_orig]

                    # Load and save pano
                    panorama_img = download_panorama(pano_id, pano_size, zoom)
                    if panorama_img:
                        filename = str(pano_id) + '.jpg'
                        panorama_img.save(f'{streetview_path}/{filename}')
                        pano_loaded = True

        # Keep track of added panos
        if pano_added & pano_loaded:
            n_panos_added += 1
            pano_id_added = pano_id
            if n_panos_added > 1:
                break
        
        # Move to next pano
        pano_ids.pop(0)

    if n_panos_added == 0:
        print(f'No pano added for tree with id {data["id"]} and position ({lat}, {lng})')
    if n_panos_added == 1:
        print(f'Only a single pano added for tree with id {data["id"]} and position ({lat}, {lng})')
