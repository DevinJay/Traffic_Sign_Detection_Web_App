from flask import Flask,render_template,request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import pandas as pd

img_size=50

app=Flask(__name__)

model=load_model('model/Traffic_signs.model')



label_cat= pd.read_csv('Meta.csv').values
label_dict=dict(zip(label_cat[:,1],label_cat[:,5]))


def preprocess(img):
	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size,1)
	return reshaped



@app.route("/")
def index():
	return(render_template("index.html"))


@app.route("/predict",methods=["POST"])
def predict():
	message= request.get_json(force=True)
	encoded=message['image']
	decoded=base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image=Image.open(dataBytesIO)

	test_image=preprocess(image)

	prediction=model.predict(test_image)
	result=np.argmax(prediction,axis=1)[0]
	accuracy=float(np.max(prediction,axis=1)[0])

	label=label_dict[result]


	response={'prediction':{'result':label, 'accuracy':accuracy }}

	return jsonify(response)

app.run(debug=True)






