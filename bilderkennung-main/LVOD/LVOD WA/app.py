from flask import Flask, render_template, Response
import cv2
import numpy as np
import random
import time
import keyboard

app = Flask(__name__)
# Aufnahme
camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 fuer webcam 1
# online livestream template (proxy issues): rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp'
# local webcam: cv2.VideoCapture(0)

# Bildgröße anzeigen
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))
size = (frame_width, frame_height)
print(size)


def object_detection():
    # FPS-Variablen deklarieren
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        # Variablen für OE festlegen
        net = cv2.dnn.readNet('yolov3_320.weights', 'yolov3_320.cfg')
    
        classes_yolo320_yolo320 = []
        with open("classes.txt", "r") as f:
            classes_yolo320 = f.read().splitlines()
        
        schriftart = cv2.FONT_HERSHEY_DUPLEX
        colors = np.random.uniform(0, 255, size=(100, 3))
        
        # Video output regeln
        savevid = cv2.VideoWriter("output.avi",
                                  cv2.VideoWriter_fourcc(*"MJPG"),
                                  2, size)
        
        
        # Objekterkennung durchführen    
        _, img= camera.read()
        height, width, _ = img.shape
        ret, frame = camera.read()
    
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
    
        boxes = []
        confidences = []
        class_ids = []
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
    
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    
        if len(indexes)!=0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes_yolo320[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), schriftart, 2, (255,255,255), 2)
        # FPS-Anzeige
        fpsfont = cv2.FONT_HERSHEY_SIMPLEX
        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(img, fps, (7,30), fpsfont, 1, (100,255,0), 3, cv2.LINE_AA)

        # Als HTML-Anzeigbare Videodatei übermitteln lassen (yield)
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', img)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')  # concat frame one by one and show result


@app.route('/output_video')
def output_video():
    #Video Stream Route ausgeben
    return Response(object_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # Homepage
    return render_template('index.html')

# Run
if __name__ == '__main__':
    app.run('localhost', 4449)

