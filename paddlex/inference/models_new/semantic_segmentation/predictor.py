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

from typing import Any, Union, Dict, List, Tuple
import numpy as np

from ....utils.func_register import FuncRegister
from ....modules.semantic_segmentation.model_list import MODELS
from ...common.batch_sampler import ImageBatchSampler
from ...common.reader import ReadImage
from ..common import (
    ResizeByShort,
    Normalize,
    ToCHWImage,
    ToBatch,
    StaticInfer,
)
from .processors import Resize, SegPostProcess
from ..base import BasicPredictor
from .result import SegResult


class SegPredictor(BasicPredictor):
    """SegPredictor that inherits from BasicPredictor."""

    entities = MODELS

    _FUNC_MAP = {}
    register = FuncRegister(_FUNC_MAP)

    def __init__(
        self,
        target_size: Union[int, Tuple[int], None] = None,
        *args: List,
        **kwargs: Dict,
    ) -> None:
        """Initializes SegPredictor.

        Args:
            target_size: Image size used for inference.
            *args: Arbitrary positional arguments passed to the superclass.
            **kwargs: Arbitrary keyword arguments passed to the superclass.
        """
        super().__init__(*args, **kwargs)
        self.target_size = target_size
        self.preprocessors, self.infer, self.postprocessers = self._build()

    def _build_batch_sampler(self) -> ImageBatchSampler:
        """Builds and returns an ImageBatchSampler instance.

        Returns:
            ImageBatchSampler: An instance of ImageBatchSampler.
        """
        return ImageBatchSampler()

    def _get_result_class(self) -> type:
        """Returns the result class, SegResult.

        Returns:
            type: The SegResult class.
        """
        return SegResult

    def _build(self) -> Tuple:
        """Build the preprocessors, inference engine, and postprocessors based on the configuration.

        Returns:
            tuple: A tuple containing the preprocessors, inference engine, and postprocessors.
        """
        preprocessors = {"Read": ReadImage(format="RGB")}
        preprocessors["ToCHW"] = ToCHWImage()
        for cfg in self.config["Deploy"]["transforms"]:
            tf_key = cfg.pop("type")
            func = self._FUNC_MAP[tf_key]
            args = cfg
            name, op = func(self, **args) if args else func(self)
            preprocessors[name] = op
        preprocessors["ToBatch"] = ToBatch()
        if "Resize" not in preprocessors:
            _, op = self._FUNC_MAP["Resize"](self, target_size=-1)
            preprocessors["Resize"] = op

        if self.target_size is not None:
            _, op = self._FUNC_MAP["Resize"](self, target_size=self.target_size)
            preprocessors["Resize"] = op

        infer = StaticInfer(
            model_dir=self.model_dir,
            model_prefix=self.MODEL_FILE_PREFIX,
            option=self.pp_option,
        )

        postprocessers = SegPostProcess()

        return preprocessors, infer, postprocessers

    def process(
        self,
        batch_data: List[Union[str, np.ndarray]],
        target_size: Union[int, Tuple[int], None] = None,
    ) -> Dict[str, Any]:
        """
        Process a batch of data through the preprocessing, inference, and postprocessing.

        Args:
            batch_data (List[Union[str, np.ndarray], ...]): A batch of input data (e.g., image file paths).
            target_size: Image size used for inference.

        Returns:
            dict: A dictionary containing the input path, raw image, and predicted segmentation maps for every instance of the batch. Keys include 'input_path', 'input_img', and 'pred'.
        """
        batch_raw_imgs = self.preprocessors["Read"](imgs=batch_data)
        batch_imgs = self.preprocessors["Resize"](
            imgs=batch_raw_imgs, target_size=target_size
        )
        batch_imgs = self.preprocessors["ToCHW"](imgs=batch_imgs)
        batch_imgs = self.preprocessors["Normalize"](imgs=batch_imgs)
        x = self.preprocessors["ToBatch"](imgs=batch_imgs)
        batch_preds = self.infer(x=x)
        if len(batch_data) > 1:
            batch_preds = np.split(batch_preds[0], len(batch_data), axis=0)

        # postprocess
        src_image_sizes = [image.shape[:2][::-1] for image in batch_raw_imgs]
        batch_preds = self.postprocessers(batch_preds, src_image_sizes)

        return {
            "input_path": batch_data,
            "input_img": batch_raw_imgs,
            "pred": batch_preds,
        }

    @register("Normalize")
    def build_normalize(
        self,
        mean=0.5,
        std=0.5,
    ):
        op = Normalize(mean=mean, std=std)
        return "Normalize", op

    @register("Resize")
    def build_resize(
        self,
        target_size=-1,
        keep_ratio=True,
        size_divisor=32,
        interp="LINEAR",
    ):
        op = Resize(
            target_size=target_size,
            keep_ratio=keep_ratio,
            size_divisor=size_divisor,
            interp=interp,
        )
        return "Resize", op