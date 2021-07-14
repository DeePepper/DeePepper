"""
    colab current path = /content/DeePepper/
    사용 방법:
    1. model을 불러오는 코드와 load_faceswap.py 같은 경로에 두기
    2. import load_faceswap
    3. model = load_faceswap.model
"""
import os
import sys
import inspect
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras.utils import conv_utils

# Custom PixelShuffler Layer
class PixelShuffler(Layer):
    """ PixelShuffler layer for Keras.
    """
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs, **kwargs):  # pylint:disable=unused-argument
        input_shape = K.int_shape(inputs)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            batch_size, channels, height, width = input_shape
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = K.reshape(inputs, (batch_size, r_height, r_width, o_channels, height, width))
            out = K.permute_dimensions(out, (0, 3, 4, 1, 5, 2))
            out = K.reshape(out, (batch_size, o_channels, o_height, o_width))
        elif self.data_format == 'channels_last':
            batch_size, height, width, channels = input_shape
            if batch_size is None:
                batch_size = -1
            r_height, r_width = self.size
            o_height, o_width = height * r_height, width * r_width
            o_channels = channels // (r_height * r_width)

            out = K.reshape(inputs, (batch_size, height, width, r_height, r_width, o_channels))
            out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            out = K.reshape(out, (batch_size, o_height, o_width, o_channels))
        return out

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank ' +
                             str(4) +
                             '; Received input shape:', str(input_shape))

        if self.data_format == 'channels_first':
            height = None
            width = None
            if input_shape[2] is not None:
                height = input_shape[2] * self.size[0]
            if input_shape[3] is not None:
                width = input_shape[3] * self.size[1]
            channels = input_shape[1] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[1]:
                raise ValueError('channels of input and size are incompatible')

            retval = (input_shape[0],
                      channels,
                      height,
                      width)
        elif self.data_format == 'channels_last':
            height = None
            width = None
            if input_shape[1] is not None:
                height = input_shape[1] * self.size[0]
            if input_shape[2] is not None:
                width = input_shape[2] * self.size[1]
            channels = input_shape[3] // self.size[0] // self.size[1]

            if channels * self.size[0] * self.size[1] != input_shape[3]:
                raise ValueError('channels of input and size are incompatible')

            retval = (input_shape[0],
                      height,
                      width,
                      channels)
        return retval

    def get_config(self):
        config = {'size': self.size,
                  'data_format': self.data_format}
        base_config = super(PixelShuffler, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

# Update layers into Keras custom objects
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj) and obj.__module__ == __name__:
        get_custom_objects().update({name: obj})


os.system('bash download.sh faceswap-model') # shell script 실행
path = os.getcwd()+"/model_dir/original.h5"
model = load_model(path, compile=False)