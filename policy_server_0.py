import os
from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/file', methods = ['POST'])
def upload_file():
    state_str = request.form['state']
    print(state_str)
    state_list = json.loads(state_str)
    print(state_list)
    upload_folder = '/home/absin/Documents/images'
    file = request.files['file']
    print(file.filename)
    file.save(os.path.join(upload_folder, file.filename))
    return jsonify({'success': True, 'body': {}})
        
if __name__ == '__main__':
    app.run()