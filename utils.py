from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import product

import cv2
import numpy as np
import requests
import vec_geohash


def download_image(tuple_xy, z):
    x, y = tuple_xy
    url = f"https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    img = requests.get(url).content
    nparr = np.frombuffer(img, np.uint8)
    bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return bgr_img


def extract_image_from_bbox(bbox, zoom):
    tiles = vec_geohash.lat_lon_bounds_to_tile_range(np.array([bbox]), zoom=16)[0]
    tiles_x = list(range(tiles[0], tiles[2] + 1))
    tiles_y = list(range(tiles[1], tiles[3] + 1))
    pixel_x_offset = tiles[0] * 256
    pixel_y_offset = tiles[1] * 256

    download_image_zoom = partial(download_image, z=zoom)
    with ThreadPoolExecutor(50) as executor:
        result = list(executor.map(download_image_zoom, product(tiles_x, tiles_y)))

    # since we downloaded images in parallel we need to reconstruct the image
    # first we stack it vertically, then we stack it horizontally after we transform it to rgb
    hight_pixels = len(tiles_y) * 256
    number_of_x_tiles = len(tiles_x)
    image = np.array(result).reshape(number_of_x_tiles, hight_pixels, 256, 3)
    image = cv2.cvtColor(np.hstack(image), cv2.COLOR_BGR2RGB)
    return image, pixel_x_offset, pixel_y_offset


def clean_mask_from_small_areas(mask, kernel_size=(15, 15)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def regularize_shapes(shape, epsilon=0.02):
    epsilon_val = epsilon * cv2.arcLength(shape, True)
    return cv2.approxPolyDP(shape, epsilon_val, closed=True)


def get_mask_shapes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [np.squeeze(regularize_shapes(shape), axis=1) for shape in contours]


def get_polygons(mask):
    mask = mask.astype("uint8")
    mask = clean_mask_from_small_areas(mask)
    polygons = get_mask_shapes(mask)
    return polygons


def pixel_to_lat_lon_polygon(polygon, zoom, append_first=True):
    polygon = vec_geohash.pixel_to_lat_lon_tuple(polygon[:, 0], polygon[:, 1], zoom=zoom)
    return np.concatenate((polygon, [polygon[0]])) if append_first else polygon


def model_find_polygons(model, bbox):
    img, x_offset, y_offset = extract_image_from_bbox(bbox, zoom=16)

    img = np.expand_dims(np.array(img / 255), axis=0)
    mask = model.predict(img)[0]
    binary_mask = (mask >= 0.5).astype(np.uint8)
    binary_mask = np.squeeze(binary_mask, axis=-1)

    return [pixel_to_lat_lon_polygon(p + [x_offset, y_offset], zoom=16) for p in get_polygons(binary_mask)]
