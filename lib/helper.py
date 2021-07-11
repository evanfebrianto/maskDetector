
import math
from typing import List, Optional, Tuple, Union

import cv2
import dataclasses
import numpy as np

from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2

import configparser
import json, re

RGB_CHANNELS = 3

def get_bb(
    image: np.ndarray,
    detection: detection_pb2.Detection):
  """Get detection bounding box in cx_min, cy_min, cx_max, cy_max

  Args:
    image: A three channel RGB image represented as numpy ndarray.
    detection: A detection proto message to be annotated on the image.

  Raises:
    ValueError: If one of the followings:
      a) If the input image is not three channel RGB.
      b) If the location data is not relative data.
  """
  if not detection.location_data:
    return
  if image.shape[2] != RGB_CHANNELS:
    raise ValueError('Input image must contain three channel rgb data.')
  image_rows, image_cols, _ = image.shape

  location = detection.location_data
  if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
    raise ValueError(
        'LocationData must be relative for this drawing funtion to work.')
  # Return bounding box if exists.
  if not location.HasField('relative_bounding_box'):
    return
  relative_bounding_box = location.relative_bounding_box
  rect_start_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
      image_rows)
  rect_end_point = _normalized_to_pixel_coordinates(
      relative_bounding_box.xmin + relative_bounding_box.width,
      relative_bounding_box.ymin + +relative_bounding_box.height, image_cols,
      image_rows)
  if rect_start_point != None and rect_end_point != None:
    return rect_start_point[0], rect_start_point[1], rect_end_point[0], rect_end_point[1]
  else:
    return None, None, None, None


def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px


def parse_config(filename='../config.ini'):
  config = configparser.ConfigParser()
  config.read(filename)

  parsed = {}
  for section_name in config.sections():
    parsed[section_name] = {}
    for option in config.options(section_name):
      option = option.upper()
      if section_name == 'TUPLE':
        parsed[section_name][option] = tuple(int(v) for v in re.findall("[0-9]+", config[section_name][option]))
      elif section_name == 'LIST':
        parsed[section_name][option] = json.loads(config[section_name][option])
      elif section_name == 'FLOAT':
        parsed[section_name][option] = float(config[section_name][option])
      elif section_name == 'INT':
        parsed[section_name][option] = int(config[section_name][option])
      elif section_name == 'BOOLEAN':
        parsed[section_name][option] = bool(int(config[section_name][option]))
      elif section_name == 'STRING':
        if option != 'RECEIVER_EMAILS':
          parsed[section_name][option] = config[section_name][option]
        else:
          emails_raw = config[section_name][option].split(',')
          emails_list = []
          for email in emails_raw:
            emails_list.append(email.strip())
          parsed[section_name][option] = emails_list
  return parsed