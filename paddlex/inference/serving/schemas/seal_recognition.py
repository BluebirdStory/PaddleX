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

from typing import Final, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal

from ..infra.models import DataInfo, PrimaryOperations
from .shared import ocr

__all__ = [
    "INFER_ENDPOINT",
    "InferRequest",
    "SealRecResult",
    "InferResult",
    "PRIMARY_OPERATIONS",
]

INFER_ENDPOINT: Final[str] = "/seal-recognition"


class InferRequest(ocr.BaseInferRequest):
    useDocOrientationClassify: Optional[bool] = None
    useDocUnwarping: Optional[bool] = None
    useLayoutDetection: Optional[bool] = None
    layoutThreshold: Optional[float] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[
        Union[float, Annotated[List[float], Field(min_length=2, max_length=2)]]
    ] = None
    layoutMergeBboxesMode: Optional[Literal["union", "large", "small"]] = None
    sealDetLimitSideLen: Optional[int] = None
    sealDetLimitType: Optional[Literal["min", "max"]] = None
    sealDetThresh: Optional[float] = None
    sealDetBoxThresh: Optional[float] = None
    sealDetUnclipRatio: Optional[float] = None
    sealRecScoreThresh: Optional[float] = None


class SealRecResult(BaseModel):
    prunedResult: dict
    sealRecImage: Optional[str] = None
    layoutDetImage: Optional[str] = None
    docPreprocessingImage: Optional[str] = None
    inputImage: Optional[str] = None


class InferResult(BaseModel):
    sealRecResults: List[SealRecResult]
    dataInfo: DataInfo


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}