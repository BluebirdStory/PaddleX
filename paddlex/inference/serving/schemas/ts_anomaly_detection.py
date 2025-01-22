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

from typing import Final

from pydantic import BaseModel

from ..infra.models import PrimaryOperations

__all__ = ["INFER_ENDPOINT", "InferRequest", "InferResult", "PRIMARY_OPERATIONS"]

INFER_ENDPOINT: Final[str] = "/time-series-anomaly-detection"


class InferRequest(BaseModel):
    csv: str


class InferResult(BaseModel):
    csv: str


PRIMARY_OPERATIONS: Final[PrimaryOperations] = {
    "infer": (INFER_ENDPOINT, InferRequest, InferResult),
}