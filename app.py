import numpy as np
import torch
import cv2
from bisenetv1 import BiSeNetV1
from utils import pre_process, post_process, direct, get_prediction, prediction_thread
from flask import Flask, jsonify, request
from threading import Thread

app = Flask(__name__)

model = BiSeNetV1(19)
model.load_state_dict(torch.load('./res/model_final_v1.pth', map_location=torch.device('cpu')))
model.cuda()
model.eval()

last = np.array([320, 320, 320]).astype(np.float32)
last = torch.from_numpy(last).cuda()
url = ""


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict()):
    if request.method == 'POST':
        vs = cv2.VideoCapture(url)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("out.avi", fourcc, 30, 
                            (640, 320), True)
        if url is not None:
            _, img = vs.read() 
            input_tensor = pre_process(img)
            out_tensor = get_prediction(model, input_tensor)
            img, last = post_process(img, out_tensor, last)

            return jsonify({'class_id': class_id, 'class_name': class_name})
        return jsonify({"url": url})


if __name__ == '__main__':
    thread = Thread(target=prediction_thread, args=(url, model, last))
    thread.daemon = True
    thread.start()
    app.run()