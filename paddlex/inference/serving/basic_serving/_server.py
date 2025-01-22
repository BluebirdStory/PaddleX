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

import logging

import uvicorn
from fastapi import FastAPI


def run_server(app: FastAPI, *, host: str, port: int, debug: bool) -> None:
    # XXX: Currently, `debug` is not used.
    # HACK: Fix duplicate logs
    uvicorn_version = tuple(int(x) for x in uvicorn.__version__.split("."))
    if uvicorn_version < (0, 19, 0):
        logging.getLogger("uvicorn").propagate = False

    # HACK
    # https://github.com/encode/starlette/issues/864
    class _EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("/health") == -1

    logging.getLogger("uvicorn.access").addFilter(_EndpointFilter())

    uvicorn.run(app, host=host, port=port, log_level="info")