import os
import requests

http_url = 'http://127.0.0.1:5003'

def inference(file_path):
    files = {}

    if not os.path.exists(file_path):
        return None
    # In (FileName, DataStream, MimeType) format?
    files['image'] = (os.path.basename(file_path), open(file_path, 'rb'), 'image/jpeg')
    response = requests.post(http_url+'/inference', files=files)
    result = response.json()
    result['httpcode'] = response.status_code

    # if 'result' in result:
    #     return result['result']
    # else:
    #     return None
    return result


if __name__ == '__main__':
    file_path = '/home/archie/Projects/MeterPose/work/pose/two_point/data/pred/byqywb_4_39_4.jpg'
    for i in range(100):
        print(inference(file_path))
