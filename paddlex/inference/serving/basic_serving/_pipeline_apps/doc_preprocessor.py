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

from fastapi import FastAPI

from ...infra import utils as serving_utils
from ...infra.config import AppConfig
from ...infra.models import ResultResponse
from ...schemas.doc_preprocessor import INFER_ENDPOINT, InferRequest, InferResult
from .._app import create_app, primary_operation
from ._common import common
from ._common import ocr as ocr_common


def create_pipeline_app(pipeline: Any, app_config: AppConfig) -> FastAPI:
    app, ctx = create_app(
        pipeline=pipeline, app_config=app_config, app_aiohttp_session=True
    )

    ocr_common.update_app_context(ctx)

    @primary_operation(
        app,
        INFER_ENDPOINT,
        "infer",
    )
    async def _infer(request: InferRequest) -> ResultResponse[InferResult]:
        pipeline = ctx.pipeline

        log_id = serving_utils.generate_log_id()

        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
        )

        doc_pp_results: List[Dict[str, Any]] = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item.json["res"])
            if ctx.config.visualize:
                output_imgs = item.img
                imgs = {
                    "input_img": img,
                    "doc_preprocessing_img": output_imgs["preprocessed_img"],
                }
                imgs = await serving_utils.call_async(
                    common.postprocess_images,
                    imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                    return_urls=ctx.extra["return_img_urls"],
                    max_img_size=ctx.extra["max_output_img_size"],
                )
            else:
                imgs = {}
            doc_pp_results.append(
                dict(
                    prunedResult=pruned_res,
                    docPreprocessingImage=imgs.get("doc_preprocessing_img"),
                    inputImage=imgs.get("input_img"),
                )
            )

        return ResultResponse[InferResult](
            logId=log_id,
            result=InferResult(
                docPreprocessingResults=doc_pp_results,
                dataInfo=data_info,
            ),
        )

    return app
