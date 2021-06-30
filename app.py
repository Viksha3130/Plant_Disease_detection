
from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf
import cv2
import pyrebase

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

#database connection
from flask import Flask, render_template,request, jsonify
from flask_mysqldb import MySQL,MySQLdb

# Define a flask app
app = Flask(__name__)

#database connection
app.secret_key = "caircocoders-ednalan"
        
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'plantdiseasedetection'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)



# Model saved with Keras model.save()
MODEL_PATH ='pdd_model_inceptionV3.h5'

# Load trained model
model = load_model(MODEL_PATH)



def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path,target_size=(500,500))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Disease: Apple Scab"
       
    elif preds==1:
        preds="Disease: Apple Black Rot Canker"
    elif preds==2:
        preds="Disease: Apple Cedar Rust "
    elif preds==3:
        preds="Healthy: Apple Leaf"
    elif preds==4:
        preds="Healthy: Arjun Leaf"
    elif preds==5:
        preds="Disease: Arjun Leaf Spot"
    elif preds==6:
        preds="Disease: Bael Chlorosis "
    elif preds==7:
        preds="Healthy: Basil Leaf"
    elif preds==8:
        preds="Disease: Black Plum Anthracnose "
    elif preds==9:
        preds="Healthy: Black Plum Leaf "
    elif preds==10:
        preds="Disease: Blackboard Foliar Galls "
    elif preds==11:
        preds="Healthy: Blackboard"
    elif preds==12:
        preds="Healthy: Blueberry"
    elif preds==13:
        preds="Disease: Broccoli Clubroot "
    elif preds==14:
        preds="Healthy: Broccoli"
    elif preds==15:
        preds="Disease: Cherry Powdery Mildew"
    elif preds==16:
        preds="Healthy: Cherry"
    elif preds==17:
        preds="Healthy: Chinar"
    elif preds==18:
        preds="Disease: Chinar Leaf Spot "
    elif preds==19:
        preds="Disease: Chinese Cabbage Fusarium Wilt "
    elif preds==20:
        preds="Healthy: Chinese Cabbage"
        
    elif preds==21:
        preds="Disease: Corn Cercospora Leaf Spot"
    elif preds==22:
        preds="Disease: Corn Common Rust"
    elif preds==23:
        preds="Disease: Corn Leaf Blight"
    elif preds==24:
        preds="Healthy: Corn "
    elif preds==25:
        preds="Healthy: Cotton"
    elif preds==26:
        preds="Disease: Cucumber Bacterial Wilt"
    elif preds==27:
        preds="Disease: Cucumber Thrips"
    elif preds==28:
        preds="Disease: Cucumber White Mold"
    elif preds==29:
        preds="Disease: Cucumber Mite"
    elif preds==30:
        preds="Disease: Grape Black Rot"
    elif preds==31:
        preds="Disease: Grape Esca"
    elif preds==32:
        preds="Disease: Grape Leaf Blight"
    elif preds==33:
        preds="Disease: Grape Blister Mite"
    elif preds==34:
        preds="Healthy: Grape"
    elif preds==35:
        preds="Disease: Grape Nitrogen Deficiency"
    elif preds==36:
        preds="Disease: Grape Powdery Mildew"
    elif preds==37:
        preds="Disease: Guava Wilt"
    elif preds==38:
        preds="Healthy: Guava"
    elif preds==39:
        preds="Healthy: Jatropha"
    elif preds==40:
        preds="Disease: Jatropha Leaf Spot"
    elif preds==41:
        preds="Disease: Lemon Citrus Canker"
    elif preds==42:
        preds="Disease: Lemon Citrus Leaf Miner"
    elif preds==43:
        preds="Healthy: Lemon"
    elif preds==44:
        preds="Disease: Mango Anthracnose"
    elif preds==45:
        preds="Healthy: Mango"
    elif preds==46:
        preds="Disease: Olive Anthracnose"
    elif preds==47:
        preds="Healthy: Olive"
    elif preds==48:
        preds="Disease: Olive Tree Leaf Moth"
    elif preds==49:
        preds="Disease: Orange Black Spot"
    elif preds==50:
        preds="Disease: Orange Canker"
    elif preds==51:
        preds="Disease: Orange Melanose"
    elif preds==52:
        preds="Disease: Orange Citrus Leaf Miner"
    elif preds==53:
        preds="Disease: Orange Citrus Thrips"
    elif preds==54:
        preds="Disease: Orange Citrus White Fly"
    elif preds==55:
        preds="Disease: Orange Greening"
    elif preds==56:
        preds="Healthy: Orange"
    elif preds==57:
        preds="Disease: Orange Red Scale"
    elif preds==58:
        preds="Disease: Pea Chlorotic Lesions"
    elif preds==59:
        preds="Healthy: Pea"
    elif preds==60:
        preds="Disease: Peach Bacterial Spot"
    elif preds==61:
        preds="Healthy: Peach"
    elif preds==62:
        preds="Disease: Pear Lace Bug"
    elif preds==63:
        preds="Disease: Pepper Bacterial Spot"
    elif preds==64:
        preds="Healthy: Pepper"
    elif preds==65:
        preds="Disease: Pomegranate Cercospora Spot"
    elif preds==66:
        preds="Healthy: Promegranate"
    elif preds==67:
        preds="Disease: Potato Early Blight"
    elif preds==68:
        preds="Disease: Potato Late Blight"
    elif preds==69:
        preds="Healthy: Potato"
    elif preds==70:
        preds="Healthy: Raspberry"
    elif preds==71:
        preds="Disease: Rice Bacterial Leaf Blight"
    elif preds==72:
        preds="Disease: Rice Brown Spot"
    elif preds==73:
        preds="Disease: Rice Leaf Smut"
    elif preds==74:
        preds="Healthy: Soybean"
    elif preds==75:
        preds="Disease: Squash Powdery Mildew"
    elif preds==76:
        preds="Disease: Strawberry Leaf Scroch"
    elif preds==77:
        preds="Healthy: Strawberry"
    elif preds==78:
        preds="Disease: Tomato Bacterial Spot"
    elif preds==79:
        preds="Disease: Tomato Early Blight"
    elif preds==80:
        preds="Disease: Tomato Late Blight"
    elif preds==81:
        preds="Disease: Tomato Leaf Mold"
    elif preds==82:
        preds="Disease: Tomato Septoria Leaf Spot"
    elif preds==83:
        preds="Disease: Tomato Spider Mites"
    elif preds==84:
        preds="Disease: Tomato Target Spot"
    elif preds==85:
        preds="Disease: Tomato Yellow Leaf Curl Virus"
    elif preds==86:
        preds="Disease: Tomato Brown Rugse Fruit Virus"
    elif preds==87:
        preds="Disease: Tomato Cracking"
    elif preds==88:
        preds="Healthy: Tomato"
    elif preds==89:
        preds="Disease: Tomato Leaf Miner"
    elif preds==90:
        preds="Disease: Tomato Mosaic Virus"
    elif preds==91:
        preds="Disease: Tomato Nutritional Deficiencies"
    elif preds==92:
        preds="Disease: Tomato Pith Necrosis"
    elif preds==93:
        preds="Velvet Leaf Weed"
    elif preds==94:
        preds="Disease: Watermelon Gummy Stem Blight"
    elif preds==95:
        preds="Healthy: Watermelon"
    elif preds==96:
        preds="Disease: Zucchini Gray Mold Rot"
    elif preds==97:
        preds="Healthy: Zucchini Healthy"
    elif preds==98:
        preds="Disease: Zucchini Leaf Curl"
    elif preds==99:
        preds="Unidentified"
    
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None

@app.route("/ajaxlivesearch",methods=["POST","GET"])
def ajaxlivesearch():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        search_word = request.form['query']
        print(search_word)
        if search_word == '':
            query = "SELECT * from dataset ORDER BY prediction"
            cur.execute(query)
            dataset = cur.fetchall()
        else:    
            query = "SELECT * from dataset WHERE prediction LIKE '%{}%' OR scientific_name LIKE '%{}%' OR common_name LIKE '%{}%' ORDER BY prediction DESC LIMIT 20".format(search_word,search_word,search_word)
            cur.execute(query)
            numrows = int(cur.rowcount)
            dataset = cur.fetchall()
            print(numrows)
    return jsonify({'htmlresponse': render_template('response.html', dataset=dataset, numrows=numrows)})


if __name__ == '__main__':
    app.run(host='192.168.0.103',port=8000,debug=True)
