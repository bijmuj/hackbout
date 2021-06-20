import numpy as np
import torch
import cv2
from bisenetv1 import BiSeNetV1
from utils import pre_process, post_process, direct, get_prediction
from flask import Flask, jsonify, request
from threading import Thread

app = Flask(__name__)

model = BiSeNetV1(19)
model.load_state_dict(torch.load('./res/model_final_v1.pth', map_location=torch.device('cpu')))
model.cuda()
model.eval()

last = np.array([320, 320, 320, 320]).astype(np.float32)
last = torch.from_numpy(last).cuda()

frame = 0.0
url = r"D:\code\Python\NN\hackbout\roadVideo_Trim.mp4"
vs = cv2.VideoCapture(url)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("out.avi", fourcc, 30, 
                    (640, 320), True)


def prediction_thread(model):
    global last
    global frame
    while True:
        grabbed, img = vs.read()
        if not grabbed:
            break
        img = cv2.resize(img, (640, 320))
        input_tensor = pre_process(img)
        out_tensor = get_prediction(model, input_tensor)
        img, last = post_process(img, out_tensor, last) 
        frame +=1
        writer.write(img)
    # return last
    writer.release()
    vs.release()


thread = Thread(target=prediction_thread, args=(model,), daemon=True)
thread.start()

@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    global frame
    global last
    if request.method == 'POST':
        # last = prediction_thread(model, last, offset)
        ret = last[0].cpu().numpy()
        return jsonify({"pos": int(ret), "frame": frame})


if __name__ == '__main__':
    app.run()