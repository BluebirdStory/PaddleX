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


import enum
import itertools
import cv2
import fitz
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import yaml

__all__ = [
    "ReaderType",
    "ImageReader",
    "VideoReader",
    "CSVReader",
    "PDFReader",
    "YAMLReader",
]


class ReaderType(enum.Enum):
    """ReaderType"""

    IMAGE = 1
    GENERATIVE = 2
    POINT_CLOUD = 3
    JSON = 4
    TS = 5
    PDF = 6
    YAML = 8


class _BaseReader(object):
    """_BaseReader"""

    def __init__(self, backend, **bk_args):
        super().__init__()
        if len(bk_args) == 0:
            bk_args = self.get_default_backend_args()
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def read(self, in_path):
        """read file from path"""
        raise NotImplementedError

    def get_backend(self, bk_args=None):
        """get the backend"""
        if bk_args is None:
            bk_args = self.bk_args
        return self._init_backend(self.bk_type, bk_args)

    def set_backend(self, backend, **bk_args):
        self.bk_type = backend
        self.bk_args = bk_args
        self._backend = self.get_backend()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        raise NotImplementedError

    def get_type(self):
        """get type"""
        raise NotImplementedError

    def get_default_backend_args(self):
        """get default backend arguments"""
        return {}


class PDFReader(_BaseReader):
    """PDFReader"""

    def __init__(self, backend="fitz", **bk_args):
        super().__init__(backend, **bk_args)

    def read(self, in_path):
        return self._backend.read_file(str(in_path))

    def _init_backend(self, bk_type, bk_args):
        return PDFReaderBackend(**bk_args)

    def get_type(self):
        return ReaderType.PDF


class ImageReader(_BaseReader):
    """ImageReader"""

    def __init__(self, backend="opencv", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def read(self, in_path):
        """read the image file from path"""
        arr = self._backend.read_file(str(in_path))
        return arr

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVImageReaderBackend(**bk_args)
        elif bk_type == "pil" or bk_type == "pillow":
            return PILImageReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return ReaderType.IMAGE


class _GenerativeReader(_BaseReader):
    """_GenerativeReader"""

    def get_type(self):
        """get type"""
        return ReaderType.GENERATIVE


def is_generative_reader(reader):
    """is_generative_reader"""
    return isinstance(reader, _GenerativeReader)


class VideoReader(_GenerativeReader):
    """VideoReader"""

    def __init__(
        self,
        backend="opencv",
        st_frame_id=0,
        max_num_frames=None,
        auto_close=True,
        **bk_args,
    ):
        super().__init__(backend=backend, **bk_args)
        self.st_frame_id = st_frame_id
        self.max_num_frames = max_num_frames
        self.auto_close = auto_close

    def read(self, in_path):
        """read vide file from path"""
        self._backend.set_pos(self.st_frame_id)
        gen = self._backend.read_file(str(in_path))
        if self.num_frames is not None:
            gen = itertools.islice(gen, self.num_frames)
        yield from gen
        if self.auto_close:
            self._backend.close()

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "opencv":
            return OpenCVVideoReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")


class YAMLReader(_BaseReader):

    def __init__(self, backend="PyYAML", **bk_args):
        super().__init__(backend, **bk_args)

    def read(self, in_path):
        return self._backend.read_file(str(in_path))

    def _init_backend(self, bk_type, bk_args):
        if bk_type == "PyYAML":
            return YAMLReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        return ReaderType.YAML


class _BaseReaderBackend(object):
    """_BaseReaderBackend"""

    def read_file(self, in_path):
        """read file from path"""
        raise NotImplementedError


class _ImageReaderBackend(_BaseReaderBackend):
    """_ImageReaderBackend"""

    pass


class OpenCVImageReaderBackend(_ImageReaderBackend):
    """OpenCVImageReaderBackend"""

    def __init__(self, flags=cv2.IMREAD_COLOR):
        super().__init__()
        self.flags = flags

    def read_file(self, in_path):
        """read image file from path by OpenCV"""
        return cv2.imread(in_path, flags=self.flags)


class PILImageReaderBackend(_ImageReaderBackend):
    """PILImageReaderBackend"""

    def __init__(self):
        super().__init__()

    def read_file(self, in_path):
        """read image file from path by PIL"""
        return ImageOps.exif_transpose(Image.open(in_path))


class PDFReaderBackend(_BaseReaderBackend):

    def __init__(self, rotate=0, zoom_x=2.0, zoom_y=2.0):
        super().__init__()
        self.mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)

    def read_file(self, in_path):
        images = []
        for page in fitz.open(in_path):
            pix = page.get_pixmap(matrix=self.mat, alpha=False)
            getpngdata = pix.tobytes(output="png")
            # decode as np.uint8
            image_array = np.frombuffer(getpngdata, dtype=np.uint8)
            img_cv = cv2.imdecode(image_array, cv2.IMREAD_ANYCOLOR)
            images.append(img_cv)
        return images


class _VideoReaderBackend(_BaseReaderBackend):
    """_VideoReaderBackend"""

    def set_pos(self, pos):
        """set pos"""
        raise NotImplementedError

    def close(self):
        """close io"""
        raise NotImplementedError


class OpenCVVideoReaderBackend(_VideoReaderBackend):
    """OpenCVVideoReaderBackend"""

    def __init__(self, **bk_args):
        super().__init__()
        self.cap_init_args = bk_args
        self._cap = None
        self._pos = 0
        self._max_num_frames = None

    def read_file(self, in_path):
        """read vidio file from path"""
        if self._cap is not None:
            self._cap_release()
        self._cap = self._cap_open(in_path)
        if self._pos is not None:
            self._cap_set_pos()
        return self._read_frames(self._cap)

    def _read_frames(self, cap):
        """read frames"""
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
        self._cap_release()

    def _cap_open(self, video_path):
        self._cap = cv2.VideoCapture(video_path, **self.cap_init_args)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open {video_path}")
        return self._cap

    def _cap_release(self):
        self._cap.release()

    def _cap_set_pos(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._pos)

    def set_pos(self, pos):
        self._pos = pos

    def close(self):
        if self._cap is not None:
            self._cap_release()
            self._cap = None


class CSVReader(_BaseReader):
    """CSVReader"""

    def __init__(self, backend="pandas", **bk_args):
        super().__init__(backend=backend, **bk_args)

    def read(self, in_path):
        """read the image file from path"""
        arr = self._backend.read_file(str(in_path))
        return arr

    def _init_backend(self, bk_type, bk_args):
        """init backend"""
        if bk_type == "pandas":
            return PandasCSVReaderBackend(**bk_args)
        else:
            raise ValueError("Unsupported backend type")

    def get_type(self):
        """get type"""
        return ReaderType.TS


class _CSVReaderBackend(_BaseReaderBackend):
    """_CSVReaderBackend"""

    pass


class PandasCSVReaderBackend(_CSVReaderBackend):
    """PandasCSVReaderBackend"""

    def __init__(self):
        super().__init__()

    def read_file(self, in_path):
        """read image file from path by OpenCV"""
        return pd.read_csv(in_path)


class YAMLReaderBackend(_BaseReaderBackend):

    def read_file(self, in_path, **kwargs):
        with open(in_path, "r", encoding="utf-8", **kwargs) as yaml_file:
            data = yaml.safe_load(yaml_file)
        return data