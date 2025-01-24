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

from email.mime import image
from typing import Any, Dict, Optional, Union, List, Tuple
import numpy as np
from ..base import BasePipeline
from .utils import get_sub_regions_ocr_res, sorted_layout_boxes
from ..components import convert_points_to_boxes
from .result import LayoutParsingResult
from ....utils import logging
from ...utils.pp_option import PaddlePredictorOption
from ...common.reader import ReadImage
from ...common.batch_sampler import ImageBatchSampler
from ..ocr.result import OCRResult

from ...models.object_detection.result import DetResult


class LayoutParsingPipeline(BasePipeline):
    """Layout Parsing Pipeline"""

    entities = ["layout_parsing"]

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the layout parsing pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        self.inintial_predictor(config)

        self.batch_sampler = ImageBatchSampler(batch_size=1)

        self.img_reader = ReadImage(format="BGR")

    def inintial_predictor(self, config: Dict) -> None:
        """Initializes the predictor based on the provided configuration.

        Args:
            config (Dict): A dictionary containing the configuration for the predictor.

        Returns:
            None
        """

        self.use_doc_preprocessor = config.get("use_doc_preprocessor", True)
        self.use_general_ocr = config.get("use_general_ocr", True)
        self.use_table_recognition = config.get("use_table_recognition", True)
        self.use_seal_recognition = config.get("use_seal_recognition", True)
        self.use_formula_recognition = config.get("use_formula_recognition", True)

        if self.use_doc_preprocessor:
            doc_preprocessor_config = config.get("SubPipelines", {}).get(
                "DocPreprocessor",
                {
                    "pipeline_config_error": "config error for doc_preprocessor_pipeline!"
                },
            )
            self.doc_preprocessor_pipeline = self.create_pipeline(
                doc_preprocessor_config
            )

        layout_det_config = config.get("SubModules", {}).get(
            "LayoutDetection",
            {"model_config_error": "config error for layout_det_model!"},
        )
        self.layout_det_model = self.create_model(layout_det_config)
        layout_kwargs = {}
        if (threshold := layout_det_config.get("threshold", None)) is not None:
            layout_kwargs["threshold"] = threshold
        if (layout_nms := layout_det_config.get("layout_nms", None)) is not None:
            layout_kwargs["layout_nms"] = layout_nms
        if (
            layout_unclip_ratio := layout_det_config.get("layout_unclip_ratio", None)
        ) is not None:
            layout_kwargs["layout_unclip_ratio"] = layout_unclip_ratio
        if (
            layout_merge_bboxes_mode := layout_det_config.get(
                "layout_merge_bboxes_mode", None
            )
        ) is not None:
            layout_kwargs["layout_merge_bboxes_mode"] = layout_merge_bboxes_mode
        self.layout_det_model = self.create_model(layout_det_config, **layout_kwargs)

        if self.use_general_ocr or self.use_table_recognition:
            general_ocr_config = config.get("SubPipelines", {}).get(
                "GeneralOCR",
                {"pipeline_config_error": "config error for general_ocr_pipeline!"},
            )
            self.general_ocr_pipeline = self.create_pipeline(general_ocr_config)

        if self.use_seal_recognition:
            seal_recognition_config = config.get("SubPipelines", {}).get(
                "SealRecognition",
                {
                    "pipeline_config_error": "config error for seal_recognition_pipeline!"
                },
            )
            self.seal_recognition_pipeline = self.create_pipeline(
                seal_recognition_config
            )

        if self.use_table_recognition:
            table_recognition_config = config.get("SubPipelines", {}).get(
                "TableRecognition",
                {
                    "pipeline_config_error": "config error for table_recognition_pipeline!"
                },
            )
            self.table_recognition_pipeline = self.create_pipeline(
                table_recognition_config
            )

        if self.use_formula_recognition:
            formula_recognition_config = config.get("SubPipelines", {}).get(
                "FormulaRecognition",
                {
                    "pipeline_config_error": "config error for formula_recognition_pipeline!"
                },
            )
            self.formula_recognition_pipeline = self.create_pipeline(
                formula_recognition_config
            )

        return

    def get_text_paragraphs_ocr_res(
        self, overall_ocr_res: OCRResult, layout_det_res: DetResult
    ) -> OCRResult:
        """
        Retrieves the OCR results for text paragraphs, excluding those of formulas, tables, and seals.

        Args:
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            layout_det_res (DetResult): The detection result containing the layout information of the document.

        Returns:
            OCRResult: The OCR result for text paragraphs after excluding formulas, tables, and seals.
        """
        object_boxes = []
        for box_info in layout_det_res["boxes"]:
            if box_info["label"].lower() in ["formula", "table", "seal"]:
                object_boxes.append(box_info["coordinate"])
        object_boxes = np.array(object_boxes)
        sub_regions_ocr_res = get_sub_regions_ocr_res(
            overall_ocr_res, object_boxes, flag_within=False
        )
        return sub_regions_ocr_res

    def get_layout_parsing_res(
        self,
        image: list,
        layout_det_res: DetResult,
        overall_ocr_res: OCRResult,
        table_res_list: list,
        seal_res_list: list,
        formula_res_list: list,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
    ) -> list:
        """
        Retrieves the layout parsing result based on the layout detection result, OCR result, and other recognition results.
        Args:
            image (list): The input image.
            layout_det_res (DetResult): The detection result containing the layout information of the document.
            overall_ocr_res (OCRResult): The overall OCR result containing text information.
            table_res_list (list): A list of table recognition results.
            seal_res_list (list): A list of seal recognition results.
            formula_res_list (list): A list of formula recognition results.
            text_det_limit_side_len (Optional[int], optional): The maximum side length of the text detection region. Defaults to None.
            text_det_limit_type (Optional[str], optional): The type of limit for the text detection region. Defaults to None.
            text_det_thresh (Optional[float], optional): The confidence threshold for text detection. Defaults to None.
            text_det_box_thresh (Optional[float], optional): The confidence threshold for text detection bounding boxes. Defaults to None
            text_det_unclip_ratio (Optional[float], optional): The unclip ratio for text detection. Defaults to None.
            text_rec_score_thresh (Optional[float], optional): The score threshold for text recognition. Defaults to None.
        Returns:
            list: A list of dictionaries representing the layout parsing result.
        """
        layout_parsing_res = []
        matched_ocr_dict = {}
        formula_index = 0
        table_index = 0
        seal_index = 0
        image = np.array(image)
        image_labels = ["image", "figure", "img", "fig"]
        object_boxes = []
        for object_box_idx, box_info in enumerate(layout_det_res["boxes"]):
            single_box_res = {}
            box = box_info["coordinate"]
            label = box_info["label"].lower()
            single_box_res["layout_bbox"] = box
            object_boxes.append(box)
            if label == "formula":
                single_box_res["formula"] = formula_res_list[formula_index][
                    "rec_formula"
                ]
                formula_index += 1
            elif label == "table":
                single_box_res["table"] = table_res_list[table_index]["pred_html"]
                table_index += 1
            elif label == "seal":
                single_box_res["seal"] = "".join(seal_res_list[seal_index]["rec_texts"])
                seal_index += 1
            else:
                ocr_res_in_box, matched_idxs = get_sub_regions_ocr_res(
                    overall_ocr_res, [box], return_match_idx=True
                )
                for matched_idx in matched_idxs:
                    if matched_ocr_dict.get(matched_idx, None) is None:
                        matched_ocr_dict[matched_idx] = [object_box_idx]
                    else:
                        matched_ocr_dict[matched_idx].append(object_box_idx)
                if label in image_labels:
                    x1, y1, x2, y2 = [int(i) for i in box]
                    sub_image = image[y1:y2, x1:x2, :]
                    single_box_res["image"] = sub_image
                    single_box_res[f"{label}_text"] = "\n".join(
                        ocr_res_in_box["rec_texts"]
                    )
                else:
                    single_box_res["text"] = "\n".join(ocr_res_in_box["rec_texts"])
            if single_box_res:
                layout_parsing_res.append(single_box_res)
        for layout_box_ids in matched_ocr_dict.values():
            # one ocr is matched to multiple layout boxes, split the text into multiple lines
            if len(layout_box_ids) > 1:
                for idx in layout_box_ids:
                    wht_im = np.ones(image.shape, dtype=image.dtype) * 255
                    box = layout_parsing_res[idx]["layout_bbox"]
                    x1, y1, x2, y2 = [int(i) for i in box]
                    wht_im[y1:y2, x1:x2, :] = image[y1:y2, x1:x2, :]
                    sub_ocr_res = next(
                        self.general_ocr_pipeline(
                            wht_im,
                            text_det_limit_side_len=text_det_limit_side_len,
                            text_det_limit_type=text_det_limit_type,
                            text_det_thresh=text_det_thresh,
                            text_det_box_thresh=text_det_box_thresh,
                            text_det_unclip_ratio=text_det_unclip_ratio,
                            text_rec_score_thresh=text_rec_score_thresh,
                        )
                    )
                    layout_parsing_res[idx]["text"] = "\n".join(
                        sub_ocr_res["rec_texts"]
                    )

        ocr_without_layout_boxes = get_sub_regions_ocr_res(
            overall_ocr_res, object_boxes, flag_within=False
        )

        for ocr_rec_box, ocr_rec_text in zip(
            ocr_without_layout_boxes["rec_boxes"], ocr_without_layout_boxes["rec_texts"]
        ):
            single_box_res = {}
            single_box_res["layout_bbox"] = ocr_rec_box
            single_box_res["text_without_layout"] = ocr_rec_text
            layout_parsing_res.append(single_box_res)

        layout_parsing_res = sorted_layout_boxes(layout_parsing_res, w=image.shape[1])

        return layout_parsing_res

    def check_model_settings_valid(self, input_params: Dict) -> bool:
        """
        Check if the input parameters are valid based on the initialized models.

        Args:
            input_params (Dict): A dictionary containing input parameters.

        Returns:
            bool: True if all required models are initialized according to input parameters, False otherwise.
        """

        if input_params["use_doc_preprocessor"] and not self.use_doc_preprocessor:
            logging.error(
                "Set use_doc_preprocessor, but the models for doc preprocessor are not initialized."
            )
            return False

        if input_params["use_general_ocr"] and not self.use_general_ocr:
            logging.error(
                "Set use_general_ocr, but the models for general OCR are not initialized."
            )
            return False

        if input_params["use_seal_recognition"] and not self.use_seal_recognition:
            logging.error(
                "Set use_seal_recognition, but the models for seal recognition are not initialized."
            )
            return False

        if input_params["use_table_recognition"] and not self.use_table_recognition:
            logging.error(
                "Set use_table_recognition, but the models for table recognition are not initialized."
            )
            return False

        return True

    def get_model_settings(
        self,
        use_doc_orientation_classify: Optional[bool],
        use_doc_unwarping: Optional[bool],
        use_general_ocr: Optional[bool],
        use_seal_recognition: Optional[bool],
        use_table_recognition: Optional[bool],
        use_formula_recognition: Optional[bool],
    ) -> dict:
        """
        Get the model settings based on the provided parameters or default values.

        Args:
            use_doc_orientation_classify (Optional[bool]): Whether to use document orientation classification.
            use_doc_unwarping (Optional[bool]): Whether to use document unwarping.
            use_general_ocr (Optional[bool]): Whether to use general OCR.
            use_seal_recognition (Optional[bool]): Whether to use seal recognition.
            use_table_recognition (Optional[bool]): Whether to use table recognition.

        Returns:
            dict: A dictionary containing the model settings.
        """
        if use_doc_orientation_classify is None and use_doc_unwarping is None:
            use_doc_preprocessor = self.use_doc_preprocessor
        else:
            if use_doc_orientation_classify is True or use_doc_unwarping is True:
                use_doc_preprocessor = True
            else:
                use_doc_preprocessor = False

        if use_general_ocr is None:
            use_general_ocr = self.use_general_ocr

        if use_seal_recognition is None:
            use_seal_recognition = self.use_seal_recognition

        if use_table_recognition is None:
            use_table_recognition = self.use_table_recognition

        if use_formula_recognition is None:
            use_formula_recognition = self.use_formula_recognition

        return dict(
            use_doc_preprocessor=use_doc_preprocessor,
            use_general_ocr=use_general_ocr,
            use_seal_recognition=use_seal_recognition,
            use_table_recognition=use_table_recognition,
            use_formula_recognition=use_formula_recognition,
        )

    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_doc_orientation_classify: Optional[bool] = None,
        use_doc_unwarping: Optional[bool] = None,
        use_general_ocr: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_table_recognition: Optional[bool] = None,
        use_formula_recognition: Optional[bool] = None,
        text_det_limit_side_len: Optional[int] = None,
        text_det_limit_type: Optional[str] = None,
        text_det_thresh: Optional[float] = None,
        text_det_box_thresh: Optional[float] = None,
        text_det_unclip_ratio: Optional[float] = None,
        text_rec_score_thresh: Optional[float] = None,
        seal_det_limit_side_len: Optional[int] = None,
        seal_det_limit_type: Optional[str] = None,
        seal_det_thresh: Optional[float] = None,
        seal_det_box_thresh: Optional[float] = None,
        seal_det_unclip_ratio: Optional[float] = None,
        seal_rec_score_thresh: Optional[float] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float]]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        **kwargs,
    ) -> LayoutParsingResult:
        """
        This function predicts the layout parsing result for the given input.

        Args:
            input (Union[str, list[str], np.ndarray, list[np.ndarray]]): The input image(s) or pdf(s) to be processed.
            use_doc_orientation_classify (bool): Whether to use document orientation classification.
            use_doc_unwarping (bool): Whether to use document unwarping.
            use_general_ocr (bool): Whether to use general OCR.
            use_seal_recognition (bool): Whether to use seal recognition.
            use_table_recognition (bool): Whether to use table recognition.
            **kwargs: Additional keyword arguments.

        Returns:
            LayoutParsingResult: The predicted layout parsing result.
        """

        model_settings = self.get_model_settings(
            use_doc_orientation_classify,
            use_doc_unwarping,
            use_general_ocr,
            use_seal_recognition,
            use_table_recognition,
            use_formula_recognition,
        )

        if not self.check_model_settings_valid(model_settings):
            yield {"error": "the input params for model settings are invalid!"}

        for img_id, batch_data in enumerate(self.batch_sampler(input)):
            image_array = self.img_reader(batch_data.instances)[0]

            if model_settings["use_doc_preprocessor"]:
                doc_preprocessor_res = next(
                    self.doc_preprocessor_pipeline(
                        image_array,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                    )
                )
            else:
                doc_preprocessor_res = {"output_img": image_array}

            doc_preprocessor_image = doc_preprocessor_res["output_img"]

            layout_det_res = next(
                self.layout_det_model(
                    doc_preprocessor_image,
                    threshold=layout_threshold,
                    layout_nms=layout_nms,
                    layout_unclip_ratio=layout_unclip_ratio,
                    layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                )
            )

            if (
                model_settings["use_general_ocr"]
                or model_settings["use_table_recognition"]
            ):
                overall_ocr_res = next(
                    self.general_ocr_pipeline(
                        doc_preprocessor_image,
                        text_det_limit_side_len=text_det_limit_side_len,
                        text_det_limit_type=text_det_limit_type,
                        text_det_thresh=text_det_thresh,
                        text_det_box_thresh=text_det_box_thresh,
                        text_det_unclip_ratio=text_det_unclip_ratio,
                        text_rec_score_thresh=text_rec_score_thresh,
                    )
                )
            else:
                overall_ocr_res = {}

            if model_settings["use_general_ocr"]:
                text_paragraphs_ocr_res = self.get_text_paragraphs_ocr_res(
                    overall_ocr_res, layout_det_res
                )
            else:
                text_paragraphs_ocr_res = {}

            if model_settings["use_table_recognition"]:
                table_res_all = next(
                    self.table_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        use_ocr_model=False,
                        overall_ocr_res=overall_ocr_res,
                        layout_det_res=layout_det_res,
                    )
                )
                table_res_list = table_res_all["table_res_list"]
            else:
                table_res_list = []

            if model_settings["use_seal_recognition"]:
                seal_res_all = next(
                    self.seal_recognition_pipeline(
                        doc_preprocessor_image,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_layout_detection=False,
                        layout_det_res=layout_det_res,
                        seal_det_limit_side_len=seal_det_limit_side_len,
                        seal_det_limit_type=seal_det_limit_type,
                        seal_det_thresh=seal_det_thresh,
                        seal_det_box_thresh=seal_det_box_thresh,
                        seal_det_unclip_ratio=seal_det_unclip_ratio,
                        seal_rec_score_thresh=seal_rec_score_thresh,
                    )
                )
                seal_res_list = seal_res_all["seal_res_list"]
            else:
                seal_res_list = []

            if model_settings["use_formula_recognition"]:
                formula_res_all = next(
                    self.formula_recognition_pipeline(
                        doc_preprocessor_image,
                        use_layout_detection=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        layout_det_res=layout_det_res,
                    )
                )
                formula_res_list = formula_res_all["formula_res_list"]
            else:
                formula_res_list = []

            parsing_res_list = self.get_layout_parsing_res(
                doc_preprocessor_image,
                layout_det_res=layout_det_res,
                overall_ocr_res=overall_ocr_res,
                table_res_list=table_res_list,
                seal_res_list=seal_res_list,
                formula_res_list=formula_res_list,
                text_det_limit_side_len=text_det_limit_side_len,
                text_det_limit_type=text_det_limit_type,
                text_det_thresh=text_det_thresh,
                text_det_box_thresh=text_det_box_thresh,
                text_det_unclip_ratio=text_det_unclip_ratio,
                text_rec_score_thresh=text_rec_score_thresh,
            )

            single_img_res = {
                "input_path": batch_data.input_paths[0],
                "page_index": batch_data.page_indexes[0],
                "doc_preprocessor_res": doc_preprocessor_res,
                "layout_det_res": layout_det_res,
                "overall_ocr_res": overall_ocr_res,
                "text_paragraphs_ocr_res": text_paragraphs_ocr_res,
                "table_res_list": table_res_list,
                "seal_res_list": seal_res_list,
                "formula_res_list": formula_res_list,
                "parsing_res_list": parsing_res_list,
                "model_settings": model_settings,
            }
            yield LayoutParsingResult(single_img_res)