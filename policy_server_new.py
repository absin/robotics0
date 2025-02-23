import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from PIL import Image
import numpy as np
import json

import os
from flask import Flask, request, jsonify


app = Flask(__name__)
config = _config.get_config("pi0_aloha_sim")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
print('Policy Loaded!!')


import numpy as np
from PIL import Image


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image




@app.route('/file', methods = ['POST'])
def upload_file():
    state_list = json.loads(request.form['state'])
    prompt = None
    if 'prompt' in request.form:
        prompt = request.form['prompt']
        print(prompt)
    upload_folder = '/home/ubuntu/images/'
    file = request.files['file']
    print(file.filename)
    file.save(os.path.join(upload_folder, file.filename))
    image = Image.open(upload_folder + file.filename).convert('RGB')
    image_data_i = np.array(image)
    new_image_data = convert_to_uint8(resize_with_pad(image_data_i, 224, 224))
    new_image_data_t = np.transpose(new_image_data, (2, 0, 1))
    if prompt:
        example = {'prompt': prompt,'images': { }, 'state': np.array(state_list)}
    else:
        example = {'images': { }, 'state': np.array(state_list)}
    example['images']['cam_high']  = new_image_data_t
    print(example['images']['cam_high'].shape)
    print('---')
    print(example['state'].shape)
    result = policy.infer(example)
    return jsonify({'success': True, 'body': result['actions'].tolist()})
        
if __name__ == '__main__':
    app.run(host='0.0.0.0')
