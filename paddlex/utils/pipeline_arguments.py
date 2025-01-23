# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ast import literal_eval
from pydantic import TypeAdapter, ValidationError
from functools import wraps
from typing import Dict, List, Tuple, Union, Literal, Optional


def custom_type(cli_expected_type):
    """Create validator for CLI input conversion and type checking"""

    def validator(cli_input: str) -> cli_expected_type:
        try:
            parsed = literal_eval(cli_input)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError) as exc:
            err = f"""Malformed input:
            - Input: {cli_input!r}
            - Error: {exc}"""
            raise ValueError(err) from exc

        try:
            return TypeAdapter(cli_expected_type).validate_python(parsed)
        except ValidationError as exc:
            err = f"""Invalid input type:
            - Expected: {cli_expected_type}
            - Received: {cli_input!r}
            """
            raise ValueError(err) from exc

    return validator


PIPELINE_ARGUMENTS = {
    "OCR": [
        {
            "name": "--use_doc_orientation_classify",
            "type": bool,
            "help": "Determines whether to use document orientation classification",
        },
        {
            "name": "--use_doc_unwarping",
            "type": bool,
            "help": "Determines whether to use document unwarping",
        },
        {
            "name": "--use_textline_orientation",
            "type": bool,
            "help": "Determines whether to consider text line orientation",
        },
        {
            "name": "--text_det_limit_side_len",
            "type": int,
            "help": "Sets the side length limit for text detection.",
        },
        {
            "name": "--text_det_limit_type",
            "type": str,
            "help": "Sets the limit type for text detection.",
        },
        {
            "name": "--text_det_thresh",
            "type": float,
            "help": "Sets the threshold for text detection.",
        },
        {
            "name": "--text_det_box_thresh",
            "type": float,
            "help": "Sets the box threshold for text detection.",
        },
        {
            "name": "--text_det_unclip_ratio",
            "type": float,
            "help": "Sets the unclip ratio for text detection.",
        },
        {
            "name": "--text_rec_score_thresh",
            "type": float,
            "help": "Sets the score threshold for text recognition.",
        },
    ],
    "object_detection": [
        {
            "name": "--threshold",
            "type": float,
            "help": "Sets the threshold for object detection.",
        },
    ],
    "image_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for image classification.",
        },
    ],
    "image_multilabel_classification": [
        {
            "name": "--threshold",
            "type": float,
            "help": "Sets the threshold for image multilabel classification.",
        },
    ],
    "pedestrian_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for human detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for pedestrian attribute recognition.",
        },
    ],
    "vehicle_attribute_recognition": [
        {
            "name": "--det_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle detection.",
        },
        {
            "name": "--cls_threshold",
            "type": float,
            "help": "Sets the threshold for vehicle attribute recognition.",
        },
    ],
    "table_recognition": None,
    "layout_parsing": None,
    "seal_recognition": None,
    "ts_cls": None,
    "ts_fc": None,
    "ts_ad": None,
    "formula_recognition": None,
    "instance_segmentation": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[float]),
            "help": "Sets the threshold for instance segmentation.",
        },
    ],
    "semantic_segmentation": [
        {
            "name": "--target_size",
            "type": custom_type(Optional[Union[int, Tuple[int, int], Literal[-1]]]),
            "help": "Sets the inference image resolution for semantic segmentation.",
        },
    ],
    "small_object_detection": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[Union[float, dict[int, float]]]),
            "help": "Sets the threshold for small object detection.",
        },
    ],
    "anomaly_detection": None,
    "video_classification": [
        {
            "name": "--topk",
            "type": int,
            "help": "Sets the Top-K value for video classification.",
        },
    ],
    "rotated_object_detection": [
        {
            "name": "--threshold",
            "type": custom_type(Optional[Union[float, dict[int, float]]]),
            "help": "Sets the threshold for rotated object detection.",
        },
    ],
    "open_vocabulary_detection": [
        {
            "name": "--thresholds",
            "type": custom_type(dict[str, float]),
            "help": "Sets the thresholds for open vocabulary detection.",
        },
        {
            "name": "--prompt",
            "type": str,
            "help": "Sets the prompt for open vocabulary detection.",
        },
    ],
    "open_vocabulary_segmentation": [
        {
            "name": "--prompt_type",
            "type": str,
            "help": "Sets the prompt type for open vocabulary segmentation.",
        },
        {
            "name": "--prompt",
            "type": custom_type(list[list[float]]),
            "help": "Sets the prompt for open vocabulary segmentation.",
        },
    ],
}
