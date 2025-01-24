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

from typing import Any, Dict, List

import numpy as np
import pycocotools.mask as mask_util
from fastapi import FastAPI

from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import ResultResponse
from ...schemas.open_vocabulary_segmentation import (
    INFER_ENDPOINT,
    InferRequest,
    InferResult,
)
from .._app import create_app, primary_operation


def _rle(mask: np.ndarray) -> str:
    rle_res = mask_util.encode(np.asarray(mask[..., None], order="F", dtype="uint8"))[0]
    return rle_res["counts"].decode("utf-8")


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline,
        app_config=app_config,
        app_aiohttp_session=True,
    )

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline
        aiohttp_session = ctx.aiohttp_session

        file_bytes = await serving_utils.get_raw_bytes_async(
            request.image, aiohttp_session
        )
        image = serving_utils.image_bytes_to_array(file_bytes)

        result = (
            await pipeline.infer(
                image,
                prompt=request.prompt,
                prompt_type=request.promptType,
            )
        )[0]

        rle_masks = [
            dict(rleResult=_rle(mask), size=mask.shape) for mask in result["masks"]
        ]
        mask_infos = result["mask_infos"]

        if ctx.config.visualize:
            output_image_base64 = serving_utils.base64_encode(
                serving_utils.image_to_bytes(result.img["res"])
            )
        else:
            output_image_base64 = None

        return ResultResponse[InferResult](
            logId=serving_utils.generate_log_id(),
            result=InferResult(
                masks=rle_masks, maskInfos=mask_infos, image=output_image_base64
            ),
        )

    return app