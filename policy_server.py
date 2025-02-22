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

@app.route('/file', methods = ['POST'])
def upload_file():
    state_list = json.loads(request.form['state'])
    prompt = request.form['prompt']
    upload_folder = '/home/ubuntu/images/'
    file = request.files['file']
    print(file.filename + ' -- '+prompt)
    file.save(os.path.join(upload_folder, file.filename))
    image = Image.open(upload_folder + file.filename)
    original_width, original_height = image.size
    new_height = 640
    padding_top = 80
    new_image = Image.new("RGB", (original_width, new_height), (0, 0, 0))

    new_image.paste(image, (0, padding_top))
    new_image_resized = new_image.resize((224, 224))
    new_image_resized_frame = np.array(new_image_resized.getdata())
    new_image_resized_frame_i = new_image_resized_frame.reshape((224, 224, 3))
    new_image_resized_frame_i_t = new_image_resized_frame_i.transpose(2, 1, 0)
    example = {'prompt': prompt,'images': { }, 'state': np.array(state_list)}
    example['images']['cam_high']  = np.array(new_image_resized_frame_i_t, dtype='uint8')
    print(example['images']['cam_high'].shape)
    print('---')
    print(example['state'].shape)
    result = policy.infer(example)
    return jsonify({'success': True, 'body': result['actions'].tolist()})
        
if __name__ == '__main__':
    app.run(host='0.0.0.0')
