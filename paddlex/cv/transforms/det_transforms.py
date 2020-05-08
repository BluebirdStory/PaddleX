# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from .ops import *
from .box_utils import *
import random
import os.path as osp
import numpy as np
from PIL import Image, ImageEnhance
import cv2


class Compose:
    """根据数据预处理/增强列表对输入数据进行操作。
       所有操作的输入图像流形状均是[H, W, C]，其中H为图像高，W为图像宽，C为图像通道数。

    Args:
        transforms (list): 数据预处理/增强列表。

    Raises:
        TypeError: 形参数据类型不满足需求。
        ValueError: 数据长度不匹配。
    """

    def __init__(self, transforms):
        if not isinstance(transforms, list):
            raise TypeError('The transforms must be a list!')
        if len(transforms) < 1:
            raise ValueError('The length of transforms ' + \
                            'must be equal or larger than 1!')
        self.transforms = transforms
        self.use_mixup = False
        for t in self.transforms:
            if t.__class__.__name__ == 'MixupImage':
                self.use_mixup = True

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (str/np.ndarray): 图像路径/图像np.ndarray数据。
            im_info (dict): 存储与图像相关的信息，dict中的字段如下：
                - im_id (np.ndarray): 图像序列号，形状为(1,)。
                - origin_shape (np.ndarray): 图像原始大小，形状为(2,)，
                                        origin_shape[0]为高，origin_shape[1]为宽。
                - mixup (list): list为[im, im_info, label_info]，分别对应
                                与当前图像进行mixup的图像np.ndarray数据、图像相关信息、标注框相关信息；
                                注意，当前epoch若无需进行mixup，则无该字段。
            label_info (dict): 存储与标注框相关的信息，dict中的字段如下：
                - gt_bbox (np.ndarray): 真实标注框坐标[x1, y1, x2, y2]，形状为(n, 4)，
                                   其中n代表真实标注框的个数。
                - gt_class (np.ndarray): 每个真实标注框对应的类别序号，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_score (np.ndarray): 每个真实标注框对应的混合得分，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - gt_poly (list): 每个真实标注框内的多边形分割区域，每个分割区域由点的x、y坐标组成，
                                  长度为n，其中n代表真实标注框的个数。
                - is_crowd (np.ndarray): 每个真实标注框中是否是一组对象，形状为(n, 1)，
                                    其中n代表真实标注框的个数。
                - difficult (np.ndarray): 每个真实标注框中的对象是否为难识别对象，形状为(n, 1)，
                                     其中n代表真实标注框的个数。
        Returns:
            tuple: 根据网络所需字段所组成的tuple；
                字段由transforms中的最后一个数据预处理操作决定。
        """

        def decode_image(im_file, im_info, label_info):
            if im_info is None:
                im_info = dict()
            try:
                im = cv2.imread(im_file).astype('float32')
            except:
                raise TypeError(
                   'Can\'t read The image file {}!'.format(im_file))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            # make default im_info with [h, w, 1]
            im_info['im_resize_info'] = np.array(
                [im.shape[0], im.shape[1], 1.], dtype=np.float32)
            # copy augment_shape from origin_shape
            im_info['augment_shape'] = np.array([im.shape[0],
                                                 im.shape[1]]).astype('int32')
            if not self.use_mixup:
                if 'mixup' in im_info:
                    del im_info['mixup']
            # decode mixup image
            if 'mixup' in im_info:
                im_info['mixup'] = \
                  decode_image(im_info['mixup'][0],
                               im_info['mixup'][1],
                               im_info['mixup'][2])
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)

        outputs = decode_image(im, im_info, label_info)
        im = outputs[0]
        im_info = outputs[1]
        if len(outputs) == 3:
            label_info = outputs[2]
        for op in self.transforms:
            if im is None:
                return None
            outputs = op(im, im_info, label_info)
            im = outputs[0]
        return outputs


class ResizeByShort:
    """根据图像的短边调整图像大小（resize）。

    1. 获取图像的长边和短边长度。
    2. 根据短边与short_size的比例，计算长边的目标长度，
       此时高、宽的resize比例为short_size/原图短边长度。
    3. 如果max_size>0，调整resize比例：
       如果长边的目标长度>max_size，则高、宽的resize比例为max_size/原图长边长度。
    4. 根据调整大小的比例对图像进行resize。

    Args:
        target_size (int): 短边目标长度。默认为800。
        max_size (int): 长边目标长度的最大限制。默认为1333。

     Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, short_size=800, max_size=1333):
        self.max_size = int(max_size)
        if not isinstance(short_size, int):
            raise TypeError(
                "Type of short_size is invalid. Must be Integer, now is {}".
                format(type(short_size)))
        self.short_size = short_size
        if not (isinstance(self.max_size, int)):
            raise TypeError("max_size: input type is invalid.")

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - im_resize_info (np.ndarray): resize后的图像高、resize后的图像宽、resize后的图像相对原始图的缩放比例
                                                 三者组成的np.ndarray，形状为(3,)。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("ResizeByShort: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('ResizeByShort: image is not 3-dimensional.')
        im_short_size = min(im.shape[0], im.shape[1])
        im_long_size = max(im.shape[0], im.shape[1])
        scale = float(self.short_size) / im_short_size
        if self.max_size > 0 and np.round(
                scale * im_long_size) > self.max_size:
            scale = float(self.max_size) / float(im_long_size)
        resized_width = int(round(im.shape[1] * scale))
        resized_height = int(round(im.shape[0] * scale))
        im_resize_info = [resized_height, resized_width, scale]
        im = cv2.resize(
            im, (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR)
        im_info['im_resize_info'] = np.array(im_resize_info).astype(np.float32)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class Padding:
    """将图像的长和宽padding至coarsest_stride的倍数。如输入图像为[300, 640],
       `coarest_stride`为32，则由于300不为32的倍数，因此在图像最右和最下使用0值
       进行padding，最终输出图像为[320, 640]。

    1. 如果coarsest_stride为1则直接返回。
    2. 获取图像的高H、宽W。
    3. 计算填充后图像的高H_new、宽W_new。
    4. 构建大小为(H_new, W_new, 3)像素值为0的np.ndarray，
       并将原图的np.ndarray粘贴于左上角。

    Args:
        coarsest_stride (int): 填充后的图像长、宽为该参数的倍数，默认为1。
        target_size (int|list): 填充后的图像长、宽，默认为1。
    """

    def __init__(self, coarsest_stride=1, target_size=None):
        self.coarsest_stride = coarsest_stride
        self.target_size = target_size

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
            ValueError: target_size小于原图的大小。
        """

        if self.coarsest_stride == 1 and self.target_size is None:
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("Padding: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Padding: image is not 3-dimensional.')
        im_h, im_w, im_c = im.shape[:]
        if self.coarsest_stride > 1:
            padding_im_h = int(
                np.ceil(im_h / self.coarsest_stride) * self.coarsest_stride)
            padding_im_w = int(
                np.ceil(im_w / self.coarsest_stride) * self.coarsest_stride)
        if self.target_size is not None:
            if isinstance(self.target_size, int):
                padding_im_h = self.target_size
                padding_im_w = self.target_size
            else:
                padding_im_h = self.target_size[0]
                padding_im_w = self.target_size[1]
            pad_height = padding_im_h - im_h
            pad_width = padding_im_w - im_w

            if pad_height < 0 or pad_width < 0:
                raise ValueError(
                'the size of image should be less than target_size, but the size of image ({}, {}), is larger than target_size ({}, {})'
                .format(im_w, im_h, padding_im_w, padding_im_h))
        padding_im = np.zeros((padding_im_h, padding_im_w, im_c),
                              dtype=np.float32)
        padding_im[:im_h, :im_w, :] = im
        if label_info is None:
            return (padding_im, im_info)
        else:
            return (padding_im, im_info, label_info)


class Resize:
    """调整图像大小（resize）。

    - 当目标大小（target_size）类型为int时，根据插值方式，
      将图像resize为[target_size, target_size]。
    - 当目标大小（target_size）类型为list或tuple时，根据插值方式，
      将图像resize为target_size。
    注意：当插值方式为“RANDOM”时，则随机选取一种插值方式进行resize。

    Args:
        target_size (int/list/tuple): 短边目标长度。默认为608。
        interp (str): resize的插值方式，与opencv的插值方式对应，取值范围为
            ['NEAREST', 'LINEAR', 'CUBIC', 'AREA', 'LANCZOS4', 'RANDOM']。默认为"LINEAR"。

    Raises:
        TypeError: 形参数据类型不满足需求。
        ValueError: 插值方式不在['NEAREST', 'LINEAR', 'CUBIC',
                    'AREA', 'LANCZOS4', 'RANDOM']中。
    """

    # The interpolation mode
    interp_dict = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'LANCZOS4': cv2.INTER_LANCZOS4
    }

    def __init__(self, target_size=608, interp='LINEAR'):
        self.interp = interp
        if not (interp == "RANDOM" or interp in self.interp_dict):
            raise ValueError("interp should be one of {}".format(
                self.interp_dict.keys()))
        if isinstance(target_size, list) or isinstance(target_size, tuple):
            if len(target_size) != 2:
                raise TypeError(
                    'when target is list or tuple, it should include 2 elements, but it is {}'
                    .format(target_size))
        elif not isinstance(target_size, int):
            raise TypeError(
                "Type of target_size is invalid. Must be Integer or List or tuple, now is {}"
                .format(type(target_size)))

        self.target_size = target_size

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if im_info is None:
            im_info = dict()
        if not isinstance(im, np.ndarray):
            raise TypeError("Resize: image type is not numpy.")
        if len(im.shape) != 3:
            raise ValueError('Resize: image is not 3-dimensional.')
        if self.interp == "RANDOM":
            interp = random.choice(list(self.interp_dict.keys()))
        else:
            interp = self.interp
        im = resize(im, self.target_size, self.interp_dict[interp])
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class RandomHorizontalFlip:
    """随机翻转图像、标注框、分割信息，模型训练时的数据增强操作。

    1. 随机采样一个0-1之间的小数，当小数小于水平翻转概率时，
       执行2-4步操作，否则直接返回。
    2. 水平翻转图像。
    3. 计算翻转后的真实标注框的坐标，更新label_info中的gt_bbox信息。
    4. 计算翻转后的真实分割区域的坐标，更新label_info中的gt_poly信息。

    Args:
        prob (float): 随机水平翻转的概率。默认为0.5。

    Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, prob=0.5):
        self.prob = prob
        if not isinstance(self.prob, float):
            raise TypeError("RandomHorizontalFlip: input type is invalid.")

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - gt_bbox (np.ndarray): 水平翻转后的标注框坐标[x1, y1, x2, y2]，形状为(n, 4)，
                                          其中n代表真实标注框的个数。
                       - gt_poly (list): 水平翻转后的多边形分割区域的x、y坐标，长度为n，
                                         其中n代表真实标注框的个数。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        if not isinstance(im, np.ndarray):
            raise TypeError(
                "RandomHorizontalFlip: image is not a numpy array.")
        if len(im.shape) != 3:
            raise ValueError(
                "RandomHorizontalFlip: image is not 3-dimensional.")
        if im_info is None or label_info is None:
            raise TypeError(
                'Cannot do RandomHorizontalFlip! ' +
                'Becasuse the im_info and label_info can not be None!')
        if 'augment_shape' not in im_info:
            raise TypeError('Cannot do RandomHorizontalFlip! ' + \
                            'Becasuse augment_shape is not in im_info!')
        if 'gt_bbox' not in label_info:
            raise TypeError('Cannot do RandomHorizontalFlip! ' + \
                            'Becasuse gt_bbox is not in label_info!')
        augment_shape = im_info['augment_shape']
        gt_bbox = label_info['gt_bbox']
        height = augment_shape[0]
        width = augment_shape[1]

        if np.random.uniform(0, 1) < self.prob:
            im = horizontal_flip(im)
            if gt_bbox.shape[0] == 0:
                if label_info is None:
                    return (im, im_info)
                else:
                    return (im, im_info, label_info)
            label_info['gt_bbox'] = box_horizontal_flip(gt_bbox, width)
            if 'gt_poly' in label_info and \
                    len(label_info['gt_poly']) != 0:
                label_info['gt_poly'] = segms_horizontal_flip(
                    label_info['gt_poly'], height, width)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class Normalize:
    """对图像进行标准化。

    1. 归一化图像到到区间[0.0, 1.0]。
    2. 对图像进行减均值除以标准差操作。

    Args:
        mean (list): 图像数据集的均值。默认为[0.485, 0.456, 0.406]。
        std (list): 图像数据集的标准差。默认为[0.229, 0.224, 0.225]。

    Raises:
        TypeError: 形参数据类型不满足需求。
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        if not (isinstance(self.mean, list) and isinstance(self.std, list)):
            raise TypeError("NormalizeImage: input type is invalid.")
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise TypeError('NormalizeImage: std is invalid!')

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (numnp.ndarraypy): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
        """
        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
        std = np.array(self.std)[np.newaxis, np.newaxis, :]
        im = normalize(im, mean, std)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class RandomDistort:
    """以一定的概率对图像进行随机像素内容变换，模型训练时的数据增强操作

    1. 对变换的操作顺序进行随机化操作。
    2. 按照1中的顺序以一定的概率在范围[-range, range]对图像进行随机像素内容变换。

    Args:
        brightness_range (float): 明亮度因子的范围。默认为0.5。
        brightness_prob (float): 随机调整明亮度的概率。默认为0.5。
        contrast_range (float): 对比度因子的范围。默认为0.5。
        contrast_prob (float): 随机调整对比度的概率。默认为0.5。
        saturation_range (float): 饱和度因子的范围。默认为0.5。
        saturation_prob (float): 随机调整饱和度的概率。默认为0.5。
        hue_range (int): 色调因子的范围。默认为18。
        hue_prob (float): 随机调整色调的概率。默认为0.5。
    """

    def __init__(self,
                 brightness_range=0.5,
                 brightness_prob=0.5,
                 contrast_range=0.5,
                 contrast_prob=0.5,
                 saturation_range=0.5,
                 saturation_prob=0.5,
                 hue_range=18,
                 hue_prob=0.5):
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob
        self.saturation_range = saturation_range
        self.saturation_prob = saturation_prob
        self.hue_range = hue_range
        self.hue_prob = hue_prob

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
        """
        brightness_lower = 1 - self.brightness_range
        brightness_upper = 1 + self.brightness_range
        contrast_lower = 1 - self.contrast_range
        contrast_upper = 1 + self.contrast_range
        saturation_lower = 1 - self.saturation_range
        saturation_upper = 1 + self.saturation_range
        hue_lower = -self.hue_range
        hue_upper = self.hue_range
        ops = [brightness, contrast, saturation, hue]
        random.shuffle(ops)
        params_dict = {
            'brightness': {
                'brightness_lower': brightness_lower,
                'brightness_upper': brightness_upper
            },
            'contrast': {
                'contrast_lower': contrast_lower,
                'contrast_upper': contrast_upper
            },
            'saturation': {
                'saturation_lower': saturation_lower,
                'saturation_upper': saturation_upper
            },
            'hue': {
                'hue_lower': hue_lower,
                'hue_upper': hue_upper
            }
        }
        prob_dict = {
            'brightness': self.brightness_prob,
            'contrast': self.contrast_prob,
            'saturation': self.saturation_prob,
            'hue': self.hue_prob
        }
        im = im.astype('uint8')
        im = Image.fromarray(im)
        for id in range(4):
            params = params_dict[ops[id].__name__]
            prob = prob_dict[ops[id].__name__]
            params['im'] = im
            if np.random.uniform(0, 1) < prob:
                im = ops[id](**params)
        im = np.asarray(im).astype('float32')
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class MixupImage:
    """对图像进行mixup操作,模型训练时的数据增强操作，目前仅YOLOv3模型支持该transform。

    当label_info中不存在mixup字段时，直接返回，否则进行下述操作：
    1. 从随机beta分布中抽取出随机因子factor。
    2.
        - 当factor>=1.0时，去除label_info中的mixup字段，直接返回。
        - 当factor<=0.0时，直接返回label_info中的mixup字段，并在label_info中去除该字段。
        - 其余情况，执行下述操作：
            （1）原图像乘以factor，mixup图像乘以(1-factor)，叠加2个结果。
            （2）拼接原图像标注框和mixup图像标注框。
            （3）拼接原图像标注框类别和mixup图像标注框类别。
            （4）原图像标注框混合得分乘以factor，mixup图像标注框混合得分乘以(1-factor)，叠加2个结果。
    3. 更新im_info中的augment_shape信息。

    Args:
        alpha (float): 随机beta分布的下限。默认为1.5。
        beta (float): 随机beta分布的上限。默认为1.5。
        mixup_epoch (int): 在前mixup_epoch轮使用mixup增强操作；当该参数为-1时，该策略不会生效。
            默认为-1。

    Raises:
        ValueError: 数据长度不匹配。
    """

    def __init__(self, alpha=1.5, beta=1.5, mixup_epoch=-1):
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in MixupImage")
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in MixupImage")
        self.mixup_epoch = mixup_epoch

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - augment_shape (np.ndarray): mixup后的图像高、宽二者组成的np.ndarray，形状为(2,)。
                   im_info删除的字段：
                       - mixup (list): 与当前字段进行mixup的图像相关信息。
                   label_info更新字段为：
                       - gt_bbox (np.ndarray): mixup后真实标注框坐标，形状为(n, 4)，
                                          其中n代表真实标注框的个数。
                       - gt_class (np.ndarray): mixup后每个真实标注框对应的类别序号，形状为(n, 1)，
                                           其中n代表真实标注框的个数。
                       - gt_score (np.ndarray): mixup后每个真实标注框对应的混合得分，形状为(n, 1)，
                                           其中n代表真实标注框的个数。

        Raises:
            TypeError: 形参数据类型不满足需求。
        """
        if im_info is None:
            raise TypeError('Cannot do MixupImage! ' +
                            'Becasuse the im_info can not be None!')
        if 'mixup' not in im_info:
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if im_info['epoch'] > self.mixup_epoch \
                or factor >= 1.0:
            im_info.pop('mixup')
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)
        if factor <= 0.0:
            return im_info.pop('mixup')
        im = self._mixup_img(im, im_info['mixup'][0], factor)
        if label_info is None:
            raise TypeError('Cannot do MixupImage! ' +
                            'Becasuse the label_info can not be None!')
        if 'gt_bbox' not in label_info or \
                'gt_class' not in label_info or \
                'gt_score' not in label_info:
            raise TypeError('Cannot do MixupImage! ' + \
                            'Becasuse gt_bbox/gt_class/gt_score is not in label_info!')
        gt_bbox1 = label_info['gt_bbox']
        gt_bbox2 = im_info['mixup'][2]['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = label_info['gt_class']
        gt_class2 = im_info['mixup'][2]['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)

        gt_score1 = label_info['gt_score']
        gt_score2 = im_info['mixup'][2]['gt_score']
        gt_score = np.concatenate(
            (gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        label_info['gt_bbox'] = gt_bbox
        label_info['gt_score'] = gt_score
        label_info['gt_class'] = gt_class
        im_info['augment_shape'] = np.array([im.shape[0],
                                             im.shape[1]]).astype('int32')
        im_info.pop('mixup')
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class RandomExpand:
    """随机扩张图像，模型训练时的数据增强操作。

    1. 随机选取扩张比例（扩张比例大于1时才进行扩张）。
    2. 计算扩张后图像大小。
    3. 初始化像素值为数据集均值的图像，并将原图像随机粘贴于该图像上。
    4. 根据原图像粘贴位置换算出扩张后真实标注框的位置坐标。

    Args:
        max_ratio (float): 图像扩张的最大比例。默认为4.0。
        prob (float): 随机扩张的概率。默认为0.5。
        mean (list): 图像数据集的均值（0-255）。默认为[127.5, 127.5, 127.5]。

    """

    def __init__(self, max_ratio=4., prob=0.5, mean=[127.5, 127.5, 127.5]):
        self.max_ratio = max_ratio
        self.mean = mean
        self.prob = prob

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，im_info更新字段为：
                       - augment_shape (np.ndarray): 扩张后的图像高、宽二者组成的np.ndarray，形状为(2,)。
                   label_info更新字段为：
                       - gt_bbox (np.ndarray): 随机扩张后真实标注框坐标，形状为(n, 4)，
                                          其中n代表真实标注框的个数。
                       - gt_class (np.ndarray): 随机扩张后每个真实标注框对应的类别序号，形状为(n, 1)，
                                           其中n代表真实标注框的个数。

        Raises:
            TypeError: 形参数据类型不满足需求。
        """
        if im_info is None or label_info is None:
            raise TypeError(
                'Cannot do RandomExpand! ' +
                'Becasuse the im_info and label_info can not be None!')
        if 'augment_shape' not in im_info:
            raise TypeError('Cannot do RandomExpand! ' + \
                            'Becasuse augment_shape is not in im_info!')
        if 'gt_bbox' not in label_info or \
                'gt_class' not in label_info:
            raise TypeError('Cannot do RandomExpand! ' + \
                            'Becasuse gt_bbox/gt_class is not in label_info!')
        prob = np.random.uniform(0, 1)
        augment_shape = im_info['augment_shape']
        im_width = augment_shape[1]
        im_height = augment_shape[0]
        gt_bbox = label_info['gt_bbox']
        gt_class = label_info['gt_class']

        if prob < self.prob:
            if self.max_ratio - 1 >= 0.01:
                expand_ratio = np.random.uniform(1, self.max_ratio)
                height = int(im_height * expand_ratio)
                width = int(im_width * expand_ratio)
                h_off = math.floor(np.random.uniform(0, height - im_height))
                w_off = math.floor(np.random.uniform(0, width - im_width))
                expand_bbox = [
                    -w_off / im_width, -h_off / im_height,
                    (width - w_off) / im_width, (height - h_off) / im_height
                ]
                expand_im = np.ones((height, width, 3))
                expand_im = np.uint8(expand_im * np.squeeze(self.mean))
                expand_im = Image.fromarray(expand_im)
                im = im.astype('uint8')
                im = Image.fromarray(im)
                expand_im.paste(im, (int(w_off), int(h_off)))
                expand_im = np.asarray(expand_im)
                for i in range(gt_bbox.shape[0]):
                    gt_bbox[i][0] = gt_bbox[i][0] / im_width
                    gt_bbox[i][1] = gt_bbox[i][1] / im_height
                    gt_bbox[i][2] = gt_bbox[i][2] / im_width
                    gt_bbox[i][3] = gt_bbox[i][3] / im_height
                gt_bbox, gt_class, _ = filter_and_process(
                    expand_bbox, gt_bbox, gt_class)
                for i in range(gt_bbox.shape[0]):
                    gt_bbox[i][0] = gt_bbox[i][0] * width
                    gt_bbox[i][1] = gt_bbox[i][1] * height
                    gt_bbox[i][2] = gt_bbox[i][2] * width
                    gt_bbox[i][3] = gt_bbox[i][3] * height
                im = expand_im.astype('float32')
                label_info['gt_bbox'] = gt_bbox
                label_info['gt_class'] = gt_class
                im_info['augment_shape'] = np.array([height,
                                                     width]).astype('int32')
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class RandomCrop:
    """随机裁剪图像。

    1. 根据batch_sampler计算获取裁剪候选区域的位置。
        (1) 根据min scale、max scale、min aspect ratio、max aspect ratio计算随机剪裁的高、宽。
        (2) 根据随机剪裁的高、宽随机选取剪裁的起始点。
        (3) 筛选出裁剪候选区域：
            - 当satisfy_all为True时，需所有真实标注框与裁剪候选区域的重叠度满足需求时，该裁剪候选区域才可保留。
            - 当satisfy_all为False时，当有一个真实标注框与裁剪候选区域的重叠度满足需求时，该裁剪候选区域就可保留。
    2. 遍历所有裁剪候选区域：
        (1) 若真实标注框与候选裁剪区域不重叠，或其中心点不在候选裁剪区域，
            则将该真实标注框去除。
        (2) 计算相对于该候选裁剪区域，真实标注框的位置，并筛选出对应的类别、混合得分。
        (3) 若avoid_no_bbox为False，返回当前裁剪后的信息即可；
            反之，要找到一个裁剪区域中真实标注框个数不为0的区域，才返回裁剪后的信息。

    Args:
        batch_sampler (list): 随机裁剪参数的多种组合，每种组合包含8个值，如下：
            - max sample (int)：满足当前组合的裁剪区域的个数上限。
            - max trial (int): 查找满足当前组合的次数。
            - min scale (float): 裁剪面积相对原面积，每条边缩短比例的最小限制。
            - max scale (float): 裁剪面积相对原面积，每条边缩短比例的最大限制。
            - min aspect ratio (float): 裁剪后短边缩放比例的最小限制。
            - max aspect ratio (float): 裁剪后短边缩放比例的最大限制。
            - min overlap (float): 真实标注框与裁剪图像重叠面积的最小限制。
            - max overlap (float): 真实标注框与裁剪图像重叠面积的最大限制。
            默认值为None，当为None时采用如下设置：
                [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                 [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
        satisfy_all (bool): 是否需要所有标注框满足条件，裁剪候选区域才保留。默认为False。
        avoid_no_bbox (bool)： 是否对裁剪图像不存在标注框的图像进行保留。默认为True。

    """

    def __init__(self,
                 batch_sampler=None,
                 satisfy_all=False,
                 avoid_no_bbox=True):
        if batch_sampler is None:
            batch_sampler = [[1, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.1, 1.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.3, 1.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.5, 1.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.7, 1.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.9, 1.0],
                             [1, 50, 0.3, 1.0, 0.5, 2.0, 0.0, 1.0]]
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当label_info为空时，返回的tuple为(im, im_info)，分别对应图像np.ndarray数据、存储与图像相关信息的字典；
                   当label_info不为空时，返回的tuple为(im, im_info, label_info)，分别对应图像np.ndarray数据、
                   存储与标注框相关信息的字典。
                   其中，label_info更新字段为：
                       - gt_bbox (np.ndarray): 随机裁剪后真实标注框坐标，形状为(n, 4)，
                                          其中n代表真实标注框的个数。
                       - gt_class (np.ndarray): 随机裁剪后每个真实标注框对应的类别序号，形状为(n, 1)，
                                           其中n代表真实标注框的个数。
                       - gt_score (np.ndarray): 随机裁剪后每个真实标注框对应的混合得分，形状为(n, 1)，
                                           其中n代表真实标注框的个数。

        Raises:
            TypeError: 形参数据类型不满足需求。
        """
        if im_info is None or label_info is None:
            raise TypeError(
                'Cannot do RandomCrop! ' +
                'Becasuse the im_info and label_info can not be None!')
        if 'augment_shape' not in im_info:
            raise TypeError('Cannot do RandomCrop! ' + \
                            'Becasuse augment_shape is not in im_info!')
        if 'gt_bbox' not in label_info or \
                'gt_class' not in label_info:
            raise TypeError('Cannot do RandomCrop! ' + \
                            'Becasuse gt_bbox/gt_class is not in label_info!')
        augment_shape = im_info['augment_shape']
        im_width = augment_shape[1]
        im_height = augment_shape[0]
        gt_bbox = label_info['gt_bbox']
        gt_bbox_tmp = gt_bbox.copy()
        for i in range(gt_bbox_tmp.shape[0]):
            gt_bbox_tmp[i][0] = gt_bbox[i][0] / im_width
            gt_bbox_tmp[i][1] = gt_bbox[i][1] / im_height
            gt_bbox_tmp[i][2] = gt_bbox[i][2] / im_width
            gt_bbox_tmp[i][3] = gt_bbox[i][3] / im_height
        gt_class = label_info['gt_class']

        gt_score = None
        if 'gt_score' in label_info:
            gt_score = label_info['gt_score']
        sampled_bbox = []
        gt_bbox_tmp = gt_bbox_tmp.tolist()
        for sampler in self.batch_sampler:
            found = 0
            for i in range(sampler[1]):
                if found >= sampler[0]:
                    break
                sample_bbox = generate_sample_bbox(sampler)
                if satisfy_sample_constraint(sampler, sample_bbox, gt_bbox_tmp,
                                             self.satisfy_all):
                    sampled_bbox.append(sample_bbox)
                    found = found + 1
        im = np.array(im)
        while sampled_bbox:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            sample_bbox = sampled_bbox.pop(idx)
            sample_bbox = clip_bbox(sample_bbox)
            crop_bbox, crop_class, crop_score = \
                filter_and_process(sample_bbox, gt_bbox_tmp, gt_class, gt_score)
            if self.avoid_no_bbox:
                if len(crop_bbox) < 1:
                    continue
            xmin = int(sample_bbox[0] * im_width)
            xmax = int(sample_bbox[2] * im_width)
            ymin = int(sample_bbox[1] * im_height)
            ymax = int(sample_bbox[3] * im_height)
            im = im[ymin:ymax, xmin:xmax]
            for i in range(crop_bbox.shape[0]):
                crop_bbox[i][0] = crop_bbox[i][0] * (xmax - xmin)
                crop_bbox[i][1] = crop_bbox[i][1] * (ymax - ymin)
                crop_bbox[i][2] = crop_bbox[i][2] * (xmax - xmin)
                crop_bbox[i][3] = crop_bbox[i][3] * (ymax - ymin)
            label_info['gt_bbox'] = crop_bbox
            label_info['gt_class'] = crop_class
            label_info['gt_score'] = crop_score
            im_info['augment_shape'] = np.array([ymax - ymin,
                                                 xmax - xmin]).astype('int32')
            if label_info is None:
                return (im, im_info)
            else:
                return (im, im_info, label_info)
        if label_info is None:
            return (im, im_info)
        else:
            return (im, im_info, label_info)


class ArrangeFasterRCNN:
    """获取FasterRCNN模型训练/验证/预测所需信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内。
    """

    def __init__(self, mode=None):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode must be in ['train', 'eval', 'test', 'quant']!")
        self.mode = mode

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当mode为'train'时，返回(im, im_resize_info, gt_bbox, gt_class, is_crowd)，分别对应
                图像np.ndarray数据、图像相当对于原图的resize信息、真实标注框、真实标注框对应的类别、真实标注框内是否是一组对象；
                当mode为'eval'时，返回(im, im_resize_info, im_id, im_shape, gt_bbox, gt_class, is_difficult)，
                分别对应图像np.ndarray数据、图像相当对于原图的resize信息、图像id、图像大小信息、真实标注框、真实标注框对应的类别、
                真实标注框是否为难识别对象；当mode为'test'或'quant'时，返回(im, im_resize_info, im_shape)，分别对应图像np.ndarray数据、
                图像相当对于原图的resize信息、图像大小信息。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        im = permute(im, False)
        if self.mode == 'train':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeFasterRCNN! ' +
                    'Becasuse the im_info and label_info can not be None!')
            if len(label_info['gt_bbox']) != len(label_info['gt_class']):
                raise ValueError("gt num mismatch: bbox and class.")
            im_resize_info = im_info['im_resize_info']
            gt_bbox = label_info['gt_bbox']
            gt_class = label_info['gt_class']
            is_crowd = label_info['is_crowd']
            outputs = (im, im_resize_info, gt_bbox, gt_class, is_crowd)
        elif self.mode == 'eval':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeFasterRCNN! ' +
                    'Becasuse the im_info and label_info can not be None!')
            im_resize_info = im_info['im_resize_info']
            im_id = im_info['im_id']
            im_shape = np.array(
                (im_info['augment_shape'][0], im_info['augment_shape'][1], 1),
                dtype=np.float32)
            gt_bbox = label_info['gt_bbox']
            gt_class = label_info['gt_class']
            is_difficult = label_info['difficult']
            outputs = (im, im_resize_info, im_id, im_shape, gt_bbox, gt_class,
                       is_difficult)
        else:
            if im_info is None:
                raise TypeError('Cannot do ArrangeFasterRCNN! ' +
                                'Becasuse the im_info can not be None!')
            im_resize_info = im_info['im_resize_info']
            im_shape = np.array(
                (im_info['augment_shape'][0], im_info['augment_shape'][1], 1),
                dtype=np.float32)
            outputs = (im, im_resize_info, im_shape)
        return outputs


class ArrangeMaskRCNN:
    """获取MaskRCNN模型训练/验证/预测所需信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内。
    """

    def __init__(self, mode=None):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode must be in ['train', 'eval', 'test', 'quant']!")
        self.mode = mode

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当mode为'train'时，返回(im, im_resize_info, gt_bbox, gt_class, is_crowd, gt_masks)，分别对应
                图像np.ndarray数据、图像相当对于原图的resize信息、真实标注框、真实标注框对应的类别、真实标注框内是否是一组对象、
                真实分割区域；当mode为'eval'时，返回(im, im_resize_info, im_id, im_shape)，分别对应图像np.ndarray数据、
                图像相当对于原图的resize信息、图像id、图像大小信息；当mode为'test'或'quant'时，返回(im, im_resize_info, im_shape)，
                分别对应图像np.ndarray数据、图像相当对于原图的resize信息、图像大小信息。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        im = permute(im, False)
        if self.mode == 'train':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeTrainMaskRCNN! ' +
                    'Becasuse the im_info and label_info can not be None!')
            if len(label_info['gt_bbox']) != len(label_info['gt_class']):
                raise ValueError("gt num mismatch: bbox and class.")
            im_resize_info = im_info['im_resize_info']
            gt_bbox = label_info['gt_bbox']
            gt_class = label_info['gt_class']
            is_crowd = label_info['is_crowd']
            assert 'gt_poly' in label_info
            segms = label_info['gt_poly']
            if len(segms) != 0:
                assert len(segms) == is_crowd.shape[0]
            gt_masks = []
            valid = True
            for i in range(len(segms)):
                segm = segms[i]
                gt_segm = []
                if is_crowd[i]:
                    gt_segm.append([[0, 0]])
                else:
                    for poly in segm:
                        if len(poly) == 0:
                            valid = False
                            break
                        gt_segm.append(np.array(poly).reshape(-1, 2))
                if (not valid) or len(gt_segm) == 0:
                    break
                gt_masks.append(gt_segm)
            outputs = (im, im_resize_info, gt_bbox, gt_class, is_crowd,
                       gt_masks)
        else:
            if im_info is None:
                raise TypeError('Cannot do ArrangeMaskRCNN! ' +
                                'Becasuse the im_info can not be None!')
            im_resize_info = im_info['im_resize_info']
            im_shape = np.array(
                (im_info['augment_shape'][0], im_info['augment_shape'][1], 1),
                dtype=np.float32)
            if self.mode == 'eval':
                im_id = im_info['im_id']
                outputs = (im, im_resize_info, im_id, im_shape)
            else:
                outputs = (im, im_resize_info, im_shape)
        return outputs


class ArrangeYOLOv3:
    """获取YOLOv3模型训练/验证/预测所需信息。

    Args:
        mode (str): 指定数据用于何种用途，取值范围为['train', 'eval', 'test', 'quant']。

    Raises:
        ValueError: mode的取值不在['train', 'eval', 'test', 'quant']之内。
    """

    def __init__(self, mode=None):
        if mode not in ['train', 'eval', 'test', 'quant']:
            raise ValueError(
                "mode must be in ['train', 'eval', 'test', 'quant']!")
        self.mode = mode

    def __call__(self, im, im_info=None, label_info=None):
        """
        Args:
            im (np.ndarray): 图像np.ndarray数据。
            im_info (dict, 可选): 存储与图像相关的信息。
            label_info (dict, 可选): 存储与标注框相关的信息。

        Returns:
            tuple: 当mode为'train'时，返回(im, gt_bbox, gt_class, gt_score, im_shape)，分别对应
                图像np.ndarray数据、真实标注框、真实标注框对应的类别、真实标注框混合得分、图像大小信息；
                当mode为'eval'时，返回(im, im_shape, im_id, gt_bbox, gt_class, difficult)，
                分别对应图像np.ndarray数据、图像大小信息、图像id、真实标注框、真实标注框对应的类别、
                真实标注框是否为难识别对象；当mode为'test'或'quant'时，返回(im, im_shape)，
                分别对应图像np.ndarray数据、图像大小信息。

        Raises:
            TypeError: 形参数据类型不满足需求。
            ValueError: 数据长度不匹配。
        """
        im = permute(im, False)
        if self.mode == 'train':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeYolov3! ' +
                    'Becasuse the im_info and label_info can not be None!')
            im_shape = im_info['augment_shape']
            if len(label_info['gt_bbox']) != len(label_info['gt_class']):
                raise ValueError("gt num mismatch: bbox and class.")
            if len(label_info['gt_bbox']) != len(label_info['gt_score']):
                raise ValueError("gt num mismatch: bbox and score.")
            gt_bbox = np.zeros((50, 4), dtype=im.dtype)
            gt_class = np.zeros((50, ), dtype=np.int32)
            gt_score = np.zeros((50, ), dtype=im.dtype)
            gt_num = min(50, len(label_info['gt_bbox']))
            if gt_num > 0:
                label_info['gt_class'][:gt_num, 0] = label_info[
                    'gt_class'][:gt_num, 0] - 1
                gt_bbox[:gt_num, :] = label_info['gt_bbox'][:gt_num, :]
                gt_class[:gt_num] = label_info['gt_class'][:gt_num, 0]
                gt_score[:gt_num] = label_info['gt_score'][:gt_num, 0]
            # parse [x1, y1, x2, y2] to [x, y, w, h]
            gt_bbox[:, 2:4] = gt_bbox[:, 2:4] - gt_bbox[:, :2]
            gt_bbox[:, :2] = gt_bbox[:, :2] + gt_bbox[:, 2:4] / 2.
            outputs = (im, gt_bbox, gt_class, gt_score, im_shape)
        elif self.mode == 'eval':
            if im_info is None or label_info is None:
                raise TypeError(
                    'Cannot do ArrangeYolov3! ' +
                    'Becasuse the im_info and label_info can not be None!')
            im_shape = im_info['augment_shape']
            if len(label_info['gt_bbox']) != len(label_info['gt_class']):
                raise ValueError("gt num mismatch: bbox and class.")
            im_id = im_info['im_id']
            gt_bbox = np.zeros((50, 4), dtype=im.dtype)
            gt_class = np.zeros((50, ), dtype=np.int32)
            difficult = np.zeros((50, ), dtype=np.int32)
            gt_num = min(50, len(label_info['gt_bbox']))
            if gt_num > 0:
                label_info['gt_class'][:gt_num, 0] = label_info[
                    'gt_class'][:gt_num, 0] - 1
                gt_bbox[:gt_num, :] = label_info['gt_bbox'][:gt_num, :]
                gt_class[:gt_num] = label_info['gt_class'][:gt_num, 0]
                difficult[:gt_num] = label_info['difficult'][:gt_num, 0]
            outputs = (im, im_shape, im_id, gt_bbox, gt_class, difficult)
        else:
            if im_info is None:
                raise TypeError('Cannot do ArrangeYolov3! ' +
                                'Becasuse the im_info can not be None!')
            im_shape = im_info['augment_shape']
            outputs = (im, im_shape)
        return outputs
