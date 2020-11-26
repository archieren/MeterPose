from flask import Flask, request
from PIL import Image
import json
import numpy as np
import sys
import traceback
from flask_tensorflow_server.python_model import python_model

app = Flask(__name__)

model = python_model(model_path='/home/archie/Projects/MeterPose/work/pose/two_point/GraphExported/tf_model_frozen.pb')

@app.route("/")
def hello():
    return "Hello!"

@app.route("/test")
def test():
    html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>object detection</title>
        </head>
        <body>
            <form action="/inference" method="post" enctype="multipart/form-data">
                <input type="file" name="image"/><br>
                <input type="submit" value="detect">
            </form>
        </body>
        </html>
    '''
    return html


@app.route('/inference', methods=['POST'])
def inference():
    result = {}

    try:
        file = request.files['image']
        # file is an werkzeug datastructure FileStorage! ‘如同’一般文件
        # print(file.filename)
        # print(file.mimetype)
        image = Image.open(file)
        # file.save('tmp_image.dat')
        # x_test = np.load('tmp_image.dat')
        output = model.inference(image)
        # print(output)

        result['ret'] = 0
        result['msg'] = 'success'
        result['result'] = output.shape
    except Exception as e:
        print('{} error {}'.format(sys._getframe().f_code.co_name, traceback.format_exc()))
        result['ret'] = 0
        result['msg'] = e.args[0]
    finally:
        print(result)
        return json.dumps(result, ensure_ascii=False, default=lambda o: o.__dict__)


if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5003)
