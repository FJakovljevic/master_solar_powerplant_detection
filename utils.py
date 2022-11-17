############################################ JSON utils
##########################################
import json

def save_as_json(json_data, path:'path where to be saved (should end with file_name.json)'):
    " function to save a json file"
    with open(path, "w") as outfile:
        json.dump(json_data, outfile)
        
def load_json_file(path:'path to to json file'):
    " function to laod a json file"
    with open(path) as infile:
        return json.load(infile)
    
    
############################################ OSM utils
##########################################
import requests

def execute_osm_json_query(query, save_json_path=None):
    from osmtogeojson import osmtogeojson
    """ 
    function that executes OSM query 
        - query: str - valid OSM query
        - save_json_path: str - if truthy it will save results as json on given path
    """
    query = '[out:json][timeout:1000];\n' + query
    req = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
    result = osmtogeojson.process_osm_json(req.json())
    if save_json_path:
        save_as_json(result, save_json_path)
    return result



############################################ MAP TILE utils
##########################################
import numpy as np

def project(lat, lon):
    ' function that projects latitude and longitude with Mercator projection to range [0,1] '
    x = 0.5 + lon/360 
    siny = np.sin((lat * np.pi)/180)
    siny = np.clip(siny, -0.9999, 0.9999)
    y = 0.5 - np.log((1+siny) / (1-siny)) / (4*np.pi)
    return x,y

def lat_lon_to_tile_vec(lat, lon, zoom):
    ' function to returne tile coordinates from lat, lat coordinates '
    scale = 1<<zoom
    x, y = project(lat, lon) 
    return np.int_(x*scale), np.int_(y*scale)

def lat_lon_to_pixel_vec(lat, lon, zoom, tile_size=256):
    ' function to return pixel coordinates from lat, lot given the zoom and tile pixel size '
    x, y = project(lat, lon)
    pixel_range = (1<<zoom)*tile_size
    return np.int_(x*pixel_range), np.int_(y*pixel_range)

def bounds_to_tiles(xmin, ymin, xmax, ymax, zoom):
    ' function to return all tiles that shape bounds intersect '
    lon, lat = np.meshgrid([xmin, xmax], [ymin, ymax])
    tiles_x, tiles_y = lat_lon_to_tile_vec(lat, lon, zoom)
    tiles_x = np.arange(np.min(tiles_x), np.max(tiles_x)+1)
    tiles_y = np.arange(np.min(tiles_y), np.max(tiles_y)+1)
    return tiles_x, tiles_y

def bounds_to_tile_tuples(xmin, ymin, xmax, ymax, zoom):
    """ 
    function to return all tiles that shape bounds intersect in tuple format
        # wrapper function to transform tiles from [[x1,x2..xn], [y1,y2..yn]] to [[x1,y1]...[xn,yn]]
    """
    tiles_x, tiles_y = bounds_to_tiles(xmin, ymin, xmax, ymax, zoom)
    return np.array(np.meshgrid(tiles_x, tiles_y)).T.reshape(-1, 2)


############################################ IMG utils
##########################################
import matplotlib.pyplot as plt
import numpy as np
import requests
import cv2

def display_img_and_masks(img, mask, predicted_mask=None):
    if predicted_mask:
        f, axarr = plt.subplots(1,3)
        axarr[0].imshow(img)
        axarr[1].imshow(mask, cmap='gray')
        axarr[2].imshow(predicted_mask, cmap='gray')
    else:
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(img)
        axarr[1].imshow(mask, cmap='gray')
    plt.show()

def download_img_from_tiles(tiles_x, tiles_y, zoom, source='arcgis'):
    img_x_stack = []
    for x in tiles_x:
        img_y_stack = []
        for y in tiles_y:
            if source == 'google':
                url = f'https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={zoom}'
            elif source == 'google_new':
                url = f'https://khms3.google.com/kh/v=932?x={x}&y={y}&z={zoom}'
            else:
                url = f'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}'
                
            img = requests.get(url).content
            nparr = np.frombuffer(img, np.uint8)
            rgb_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            img_y_stack.append(rgb_img)

        img_y = cv2.vconcat(img_y_stack)
        img_x_stack.append(img_y)

    img = cv2.hconcat(img_x_stack)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def img_spliter(image, pixel_size=256, color=True):
    ' funtion that splits image into smaller squares '
    return [image[y:y+pixel_size, x:x+pixel_size, :] if color else image[y:y+pixel_size, x:x+pixel_size]
                                                                for x in range(0, image.shape[1], pixel_size)
                                                                    for y in range(0, image.shape[0], pixel_size)]


############################################ SETUP utils
##########################################
import os

def make_train_tes_folder_structure(datase_name):
    train_img_path = rf'D:\documents\{datase_name}\train_images\img'
    os.makedirs(train_img_path)
    
    train_masks_path = rf'D:\documents\{datase_name}\train_masks\img'
    os.makedirs(train_masks_path)
    
    val_img_path = rf'D:\documents\{datase_name}\val_images\img'
    os.makedirs(val_img_path)
    
    val_masks_path = rf'D:\documents\{datase_name}\val_masks\img'
    os.makedirs(val_masks_path)
    
    return train_img_path, train_masks_path, val_img_path, val_masks_path