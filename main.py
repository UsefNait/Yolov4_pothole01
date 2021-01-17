from flask import Flask, render_template, Response ,request, session
from camera import VideoCamera
import path
import numpy as np 
import argparse
import imutils 
import time
import cv2
import os
from PIL import Image, ImageOps
from yolo_img_detection import ImageDetection
import openpyxl
from csv import writer 

app = Flask(__name__)
app.secret_key = "super secret key"
#Directory to store uploaded images
app.config["IMAGE_UPLOADS"] = "./static/uploads"


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/cam')
def Camera():
    return render_template('cam.html')

@app.route('/portfolio')
def portfolio():
    print("portfolio")
    return render_template('portfolio-details.html')

@app.route('/yolo_video')
def yolo_video():
	return render_template('yolo_video.html')

@app.route('/yolo_out_put_img')
def yolo_out_put_img():
    return render_template('yolo_out_put_img.html')

@app.route('/upload_video_yolo', methods=['GET', 'POST'])
def upload_video_yolo():
    if request.method == 'POST':
        print("upload_video_yolo")
        if 'video' in request.files:
            print("upload_video_yolo")
            user=request.form['name']
            phone=request.form['phone']

            tmp = tmp="./static/uploads"
            #data = np.ndarray(shape=(1, 1024, 1024, 3), dtype=np.float32)
            img=request.files['video']
            img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
            session['video'] = os.path.join(app.config["IMAGE_UPLOADS"], img.filename)
            session['videooutput'] = 'static/uploads/'+img.filename[:-4]+"_pred.webm"
            print("...............videooutput..........." + session['videooutput'])

            return render_template('yolo_detection_video.html')
            
    return render_template('upload_video_yolo.html')

def gen(camera):
	
    while True:
        if camera.get_frame() == -1 :
            camera.vs.release()
            print("fin.............................")
            break
        frame = camera.get_frame()
        #time.sleep(0)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(session['video'] , session['videooutput'])),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/yolo_video_output')
def yolo_video_output():

    return render_template('yolo_video_output.html')


#Test the uploaded image
@app.route('/upload_img_yolo', methods=['GET', 'POST'])
def upload_img_yolo():
    if request.method == 'POST':
        if 'image' in request.files:
            user=request.form['name']
            phone=request.form['phone']
            latitude=request.form['latitude']
            longitude=request.form['longitude']

            tmp = tmp="./static/uploads"
            data = np.ndarray(shape=(1, 1024, 1024, 3), dtype=np.float32)
            img=request.files['image']
            img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
            image=Image.open(tmp+'/'+img.filename)
            size = (1024, 1024)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            imageDetection = ImageDetection(tmp+'/'+img.filename , tmp+'/pred_'+img.filename)
            #image_pred = imageDetection.get_frame()
            nbobject = imageDetection.get_frame()
            print(str(nbobject))
            geocode = latitude, longitude
            #image_array = np.asarray(image)
            #normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            #data[0] = normalized_image_array
            img_input_output = tmp+'/'+img.filename , tmp+'/pred_'+img.filename

            field_names = ['user','phone','latitude', 'longitude', 'image']
            with open('./static/potholes.csv', 'a+', newline='') as logfile:
                #logger = csv.DictWriter(logfile, fieldnames=field_names)
                #logger.writeheader()
                detection_result = [user,phone , latitude ,longitude ,tmp+'/pred_'+img.filename ]
                ##detection_result['user'] = user
                ##detection_result['phone'] = phone
                ##detection_result['latitude'] = latitude
                ##detection_result['longitude'] = longitude
                ##detection_result['image'] = tmp+'/pred_'+img.filename
                #writer = csv.DictWriter(logfile, fieldnames=field_names)
                csv_writer = writer(logfile)
                csv_writer.writerow(detection_result)

            return render_template("yolo_out_put_img.html" , geocode=geocode , nbobject = nbobject , img_input_output = img_input_output)
    return render_template("upload_img_yolo.html")        




if __name__ == '__main__':

    ap= argparse.ArgumentParser()
    #ap.add_argument("-i", "--input", required=True,help="path to input video")
	#ap.add_argument("-o", "--output", required=True,help="path to output video")
	#ap.add_argument("-y", "--yolo", required=True,help="base path to YOLO directory")
	#ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applyong non-maxima suppression")
    #args = vars(ap.parse_args())
    #path.s = args["input"]
    
	
    app.run()	
