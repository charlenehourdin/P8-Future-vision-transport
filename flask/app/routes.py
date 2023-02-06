import numpy as np
import matplotlib.pyplot as plt
import os
from flask import render_template, request
from app import web
import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
from app.scripts.utils import pred_seg, class_colors


# Paramètres
class_name = [
    'void', 'flat', 'construction', 'object', 'nature', 'sky', 'human',
    'vehicle'
]
img_height = 128
img_width = 256
n_classes = 8


@web.route('/')
def index():
    image_list = os.listdir('./app/static/datasets/images')
    return render_template('index.html', image_list=image_list)


@web.route('/api', methods=['POST'])
def seg_infer():
    # Retourner le fichier sélectionné
    image = request.form['image']
    image_path = str('./app/static/datasets/images/' + image)
    gt_path = str('./app/static/datasets/gt/' + image)

    model = tf.keras.models.load_model('./app/models/unet_mini.h5', compile=False)

    # Chargement et traitement des masques
    gt = img_to_array(
        load_img(gt_path,
                 target_size=(img_height, img_width),
                 color_mode="grayscale"))
    gt = np.squeeze(gt)
    plt.imsave('./app/static/outputs/gt.png',
               gt,
               cmap='nipy_spectral_r')

    # Prédire à partir de l'image
    seg_img = pred_seg(model=model,
                    inp=image_path,
                    out_fname='./app/static/outputs/prediction.png',
                    n_classes=n_classes,
                    colors=class_colors,
                    prediction_width=256,
                    prediction_height=128,
                    read_image_type=1)
    plt.imsave('./app/static/outputs/prediction.png',
               seg_img,
               cmap='nipy_spectral_r')

    return render_template('api.html', image_data=image)

@web.route('/sw.js', methods=['GET'])
def sw():
    return web.send_static_file('sw.js')