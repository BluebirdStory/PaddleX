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

from typing import Final, List, Optional

from pydantic import BaseModel
from typing_extensions import Literal

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "OCRResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/ocr"


class InferRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useTextlineOrientation: Optional[bool] = False
    textDetLimitSideLen: Optional[int] = None
    textDetLimitType: Optional[Literal["min", "max"]] = None
    # Better to use "threshold"? Be consistent with the pipeline API though.
    textDetThresh: Optional[float] = None
    textDetBoxThresh: Optional[float] = None
    textDetUnclipRatio: Optional[float] = None
    textRecScoreThresh: Optional[float] = None


class OCRResult(BaseModel):
    prunedResult: dict
    ocrImage: Optional[str] = None
    docPreprocessingImage: Optional[str] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    ocrResults: List[OCRResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}