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

from typing import Any, Dict, Optional, Union, List
import pandas as pd

from ...utils.pp_option import PaddlePredictorOption
from ..base import BasePipeline

from ...models.ts_forecasting.result import TSFcResult


class TSFcPipeline(BasePipeline):
    """TSFcPipeline Pipeline"""

    entities = "ts_forecast"

    def __init__(
        self,
        config: Dict,
        device: str = None,
        pp_option: PaddlePredictorOption = None,
        use_hpip: bool = False,
    ) -> None:
        """Initializes the Time Series Forecast pipeline.

        Args:
            config (Dict): Configuration dictionary containing various settings.
            device (str, optional): Device to run the predictions on. Defaults to None.
            pp_option (PaddlePredictorOption, optional): PaddlePredictor options. Defaults to None.
            use_hpip (bool, optional): Whether to use high-performance inference (hpip) for prediction. Defaults to False.
        """

        super().__init__(device=device, pp_option=pp_option, use_hpip=use_hpip)

        ts_forecast_model_config = config["SubModules"]["TSForecast"]
        self.ts_forecast_model = self.create_model(ts_forecast_model_config)

    def predict(
        self, input: Union[str, List[str], pd.DataFrame, List[pd.DataFrame]], **kwargs
    ) -> TSFcResult:
        """Predicts time series forecast results for the given input.

        Args:
            input (Union[str, list[str], pd.DataFrame, list[pd.DataFrame]]): The input image(s) or path(s) to the images.
            **kwargs: Additional keyword arguments that can be passed to the function.

        Returns:
            TSFcResult: The predicted time series forecast results.
        """
        yield from self.ts_forecast_model(input)