import tensorflow as tf
import numpy as np
import cv2
from PanGan_pro import PanGan as PanGan_pro
import scipy.io as scio
import time
import os
import tifffile
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

PRO=1
if(PRO):
    ckpt='./'
    result="./result_pro"
else:
    ckpt='./trainDir/model_org-generator/PanNet-120000'
    result="./result_org"

'''Defining parameters'''
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('pan_size',
                           default_value=128,
                           docstring='pan image size')
tf.app.flags.DEFINE_integer('ms_size',
                           default_value=32,
                           docstring='ms image size')
tf.app.flags.DEFINE_integer('batch_size',
                           default_value=1,
                           docstring='img batch')
tf.app.flags.DEFINE_integer('num_spectrum',
                           default_value=4,
                           docstring='spectrum num')
tf.app.flags.DEFINE_integer('ratio',
                           default_value=4,
                           docstring='pan image/ms img')
tf.app.flags.DEFINE_string('model_path',
                           default_value=ckpt,

                           docstring='pan image/ms img') 
tf.app.flags.DEFINE_string('test_path',
                           default_value='./data/test_gt',
                           docstring='test img data')                            
tf.app.flags.DEFINE_string('result_path',
                           default_value=result,
                           docstring='result img')
tf.app.flags.DEFINE_string('dataType',
                           default_value="NPY",
                           docstring='test data type: [TIF, NPY, MAT]')

def read_img(pan_test_path, ms_test_path, img_name, FLAGS):
    pan_img_path = os.path.join(pan_test_path, img_name)
    ms_img_path = os.path.join(ms_test_path, img_name)
    if (FLAGS.dataType == "NPY"):
        pan_img = read8bit(pan_img_path, 'pan')
        h, w = pan_img.shape
        pan_img = pan_img.reshape((1, h, w, 1))
        ms_img = read8bit(ms_img_path, 'ms')
        h, w, c = ms_img.shape
        ms_img = cv2.resize(ms_img, (4 * w, 4 * h), interpolation=cv2.INTER_CUBIC)
        h, w, c = ms_img.shape
        ms_img = ms_img.reshape((1, h, w, c))
    elif (FLAGS.dataType == "TIF"):
        ms_img = gdal_read(ms_img_path, 'ms')
        pan_img = gdal_read(pan_img_path, 'pan')
    elif (FLAGS.dataType == "MAT"):
        ms_img = scio.loadmat(ms_img_path)['I']
        ms_img = (ms_img - 127.5) / 127.5
        pan_img = scio.loadmat(pan_img_path)['I']
        pan_img = (pan_img - 127.5) / 127.5
    return pan_img, ms_img

def gdal_read(path, name):
    data = gdal.Open(path)
    w = data.RasterXSize
    h = data.RasterYSize
    img = data.ReadAsArray(0, 0, w, h)
    if name == 'ms':
        img = np.transpose(img, (1, 2, 0))
    img = (img - 1023.5) / 1023.5
    return img

def read8bit(path, name):
    if name == 'ms':
        v = 'src'
    else:
        v = 'pan'
    v = 'I'
    img = np.load(path)
    img = (img - 127.5) / 127.5
    return img

def img_write(img_array, save_path):
    datatype = gdal.GDT_Byte
    h, w, c = img_array.shape
    driver = gdal.GetDriverByName('GTiff')
    data = driver.Create(save_path, w, h, c, datatype)
    for i in range(c):
        data.GetRasterBand(i + 1).WriteArray(img_array[:, :, i])
    del data

def main(argv):
    if not os.path.exists(FLAGS.result_path):
        os.makedirs(FLAGS.result_path)
    if(PRO):
        model=PanGan_pro(FLAGS.pan_size,FLAGS.ms_size, FLAGS.batch_size, FLAGS.num_spectrum, FLAGS.ratio,0.0001, 0.99, 1000,False)
    else:
        model=PanGan(FLAGS.pan_size,FLAGS.ms_size, FLAGS.batch_size, FLAGS.num_spectrum, FLAGS.ratio,0.0001, 0.99, 1000,False)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, FLAGS.model_path)
        ms_test_path= FLAGS.test_path + '/'
        pan_test_path=FLAGS.test_path + '/'
        for img_name in os.listdir(ms_test_path):
            start=time.time()
            print(img_name)
            pan, ms = read_img(pan_test_path, ms_test_path, img_name,FLAGS)
            start=time.time()
            if 0:
                PanSharpening,PanImg,error,error2= sess.run([model.PanSharpening_img_pan,model.pan_img_i,model.g_spectrum_loss,model.g_spatial_loss], feed_dict={model.pan_img:pan, model.ms_img:ms})
                PanSharpening=PanSharpening*127.5+127.5
                PanSharpening=PanSharpening.squeeze()
                PanSharpening=PanSharpening.astype('uint8')
                PanImg=PanImg*127.5+127.5
                PanImg=PanImg.squeeze()
                PanImg=PanImg.astype('uint8')
            else:
                PanSharpening,PanImg,error,error2= sess.run([model.PanSharpening_img,model.ms_img_i,model.g_spectrum_loss,model.g_spatial_loss], feed_dict={model.pan_img:pan, model.ms_img:ms})
                PanSharpening=PanSharpening*127.5+127.5
                PanSharpening=PanSharpening.squeeze()
                PanSharpening=PanSharpening.astype('uint8')
                PanImg=ms*127.5+127.5
                PanImg=PanImg.squeeze()
                PanImg = PanImg.astype('uint8')
                PanSharpening=cv2.resize(PanSharpening,(PanImg.shape[1], PanImg.shape[0]),interpolation=cv2.INTER_CUBIC)

            end=time.time()
            print(end-start)
            save_name=img_name.split('.')[0] + '.TIF'
            save_path=os.path.join(FLAGS.result_path,save_name)
            img_write(PanSharpening.astype('uint8')[:,:,0:4], save_path)  ##
            img_write(PanImg.astype('uint8')[:,:,0:4], save_path.replace(".TIF","_org.TIF"))

            difImg = PanImg-PanSharpening
            difOneImg=difImg[:,:,0]
            for i in range(1,difImg.shape[-1]):
                difOneImg+=difImg[:,:,i]
            difOneImg[np.where(difOneImg<150)] = 0
            tifffile.imsave(save_path.replace(".TIF","_dif.TIF"), difOneImg.astype('uint8'))
            print(img_name + ' done.' + 'spectrum error is ' + str(error) + 'spatial error is ' + str(error2))

if __name__ == '__main__':
    tf.app.run()
    
      
    
