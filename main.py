##MSDI-GAN 2022/8/13
import tensorflow as tf
import numpy as np

def Log10(input):
    return tf.math.log(input) / tf.math.log(10.)

def high_pass(self, img, type='PanSharepening'):
    if type == 'pan':
        input = img
        for i in range(self.num_spectrum - 1):
            input = tf.concat([input, img], axis=-1)
        img = input
    blur_kerel = np.zeros(shape=(13, 13, self.num_spectrum, self.num_spectrum), dtype=np.float32)
    value = 1 / 169 * np.ones(shape=(13, 13), dtype=np.float32)
    for i in range(self.num_spectrum):
        blur_kerel[:, :, i, i] = value
    img_lp = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')
    img_hp = tf.reshape(tf.reduce_mean(img - img_lp, 3), [self.batch_size, self.pan_size, self.pan_size, 1])
    return tf.abs(img_hp)

def high_pass_1(self, img, type='PanSharepening'):
    if type == 'pan':
        input = img
        for i in range(3):
            input = tf.concat([input, img], axis=-1)
        img = input
    blur_kerel = np.zeros(shape=(3, 3, 4, 4), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    for i in range(4):
        blur_kerel[:, :, i, i] = value
    img_hp = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')

    return tf.abs(img_hp)

def high_pass_2(self, img, type='PanSharepening'):
    blur_kerel = np.zeros(shape=(3, 3, 1, 1), dtype=np.float32)
    value = np.array([[1., 1., 1.], [1., -8., 1.], [1., 1., 1.]])
    blur_kerel[:, :, 0, 0] = value
    img_hp = tf.nn.conv2d(img, tf.convert_to_tensor(blur_kerel), strides=[1, 1, 1, 1], padding='SAME')
    # img_hp=tf.reshape(tf.reduce_mean(img_hp,3),[self.batch_size,128,128,1])
    # img_hp=img-img_lp
    return img_hp

def lrelu(self, x, leak=0.2):
    return tf.maximum(x, leak * x)

class PanGan(object):
    
    def __init__(self, pan_size, ms_size, batch_size,num_spectrum, ratio,init_lr=0.0001,lr_decay_rate=0.99,lr_decay_step=1000, is_training=True):
        self.num_spectrum=num_spectrum
        self.is_training=is_training
        self.ratio = ratio
        self.batch_size=batch_size
        self.pan_size=pan_size
        self.ms_size=ms_size
        self.init_lr=init_lr
        self.lr_decay_rate=lr_decay_rate
        self.lr_decay_step=lr_decay_step
        self.build_model(pan_size, ms_size, batch_size,num_spectrum, is_training)

    def build_model(self, pan_size, ms_size, batch_size, num_spectrum, is_training):

        if is_training:
            with tf.name_scope('input'):
                self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size, pan_size, pan_size, 1), name='pan_placeholder')
                self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,ms_size, ms_size, num_spectrum), name='ms_placeholder')
                self.ms_img_=tf.image.resize_images(images=self.ms_img, size=[pan_size, pan_size],method=2)
                self.pan_img_hp=self.high_pass_2(self.pan_img, 'pan')
            with tf.name_scope('PanSharpening'):

                self.PanSharpening_img,conv4,conv3,conv2,conv1= self.PanSharpening_model_dense_pro(self.pan_img, self.ms_img)

                self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[ms_size, ms_size],
                                                               method=tf.image.ResizeMethod.BILINEAR)

                self.PanSharpening_img_pan=tf.reshape(tf.reduce_mean(self.PanSharpening_img, axis=3), (batch_size, pan_size, pan_size, 1))

                self.PanSharpening_img_hp = self.high_pass_2(self.PanSharpening_img_pan)
            
            with tf.name_scope('d_loss'):
                with tf.name_scope('spatial_loss'):
                    # spatial_pos=self.spatial_discriminator(self.pan_img_4, reuse=False)
                    spatial_pos = self.spatial_discriminator(self.pan_img, reuse=False, more=None)
                    # spatial_neg=self.spatial_discriminator(self.PanSharpening_img, reuse=True)
                    spatial_neg = self.spatial_discriminator(self.PanSharpening_img_pan, reuse=True, more=[conv1,conv2,conv3,conv4])
                    spatial_pos_loss= tf.reduce_mean(tf.square(spatial_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spatial_neg_loss= tf.reduce_mean(tf.square(spatial_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spatial_loss=spatial_pos_loss + spatial_neg_loss
                    tf.summary.scalar('spatial_loss', self.spatial_loss)
                with tf.name_scope('spectrum_loss'):
                    spectrum_pos=self.spectrum_discriminator(self.ms_img_, reuse=False, more=None)
                    spectrum_neg=self.spectrum_discriminator(self.PanSharpening_img, reuse=True, more=[conv1,conv2,conv3,conv4])
                    spectrum_pos_loss= tf.reduce_mean(tf.square(spectrum_pos-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                    spectrum_neg_loss= tf.reduce_mean(tf.square(spectrum_neg-tf.zeros(shape=[batch_size,1], dtype=tf.float32)))
                    self.spectrum_loss=spectrum_pos_loss + spectrum_neg_loss
                    tf.summary.scalar('spectrum_loss', self.spectrum_loss)
            
            with tf.name_scope('g_loss'):
                spatial_loss_ad= tf.reduce_mean(tf.square(spatial_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spatial_loss_ad', spatial_loss_ad)
                spectrum_loss_ad=tf.reduce_mean(tf.square(spectrum_neg-tf.ones(shape=[batch_size,1], dtype=tf.float32)))
                tf.summary.scalar('spectrum_loss_ad', spectrum_loss_ad)
                g_spatital_loss= tf.reduce_mean(tf.square(self.PanSharpening_img_hp-self.pan_img_hp))
                tf.summary.scalar('g_spatital_loss', g_spatital_loss)
                g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img-self.ms_img_))
                tf.summary.scalar('g_spectrum_loss', g_spectrum_loss)
                self.g_loss= 0.001*spatial_loss_ad + 0.002*spectrum_loss_ad+g_spatital_loss +g_spectrum_loss
                tf.summary.scalar('g_loss', self.g_loss)
            
            with tf.name_scope('valid_error'):
                self.valid_spatital_error=tf.reduce_mean(tf.abs(self.PanSharpening_img_hp-self.pan_img_hp))
                tf.summary.scalar('valid_spatital_error', self.valid_spatital_error)
                self.valid_spectrum_error=tf.reduce_mean(tf.abs(self.PanSharpening_img-self.ms_img_))
                tf.summary.scalar('valid_spectrum_error', self.valid_spectrum_error)
                ###MSE, RMSE, PSNR
                #--Pan
                self.Mse_p = tf.reduce_mean(tf.pow(tf.abs(self.PanSharpening_img_hp-self.pan_img_hp), 2)) / self.pan_size*self.pan_size
                self.Rmse_p = tf.sqrt(self.Mse_p)
                self.Psnr_p = 10 * Log10((255*255)/(self.Mse_p))
                tf.summary.scalar('Mse_p', self.Mse_p)
                tf.summary.scalar('Rmse_p', self.Rmse_p)
                tf.summary.scalar('Psnr_p', self.Psnr_p)
                #--Ms
                self.Mse_m = tf.reduce_mean(tf.pow(tf.abs(self.PanSharpening_img-self.ms_img_), 2)) / self.pan_size*self.pan_size
                self.Rmse_m = tf.sqrt(self.Mse_m)
                self.Psnr_m = 10 * Log10((255*255)/(self.Mse_m))
                tf.summary.scalar('Mse_m', self.Mse_m)
                tf.summary.scalar('Rmse_m', self.Rmse_m)
                tf.summary.scalar('Psnr_m', self.Psnr_m)
        else:
            with tf.name_scope('input'):
                self.pan_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,None, None, 1), name='pan_placeholder')
                self.ms_img=tf.placeholder(dtype=tf.float32, shape=(batch_size,None, None, num_spectrum), name='ms_placeholder')
            ##resize
            self.pan_img_i=tf.image.resize_images(images=self.pan_img, size=[128, 128],
                                                               method=tf.image.ResizeMethod.BILINEAR)
            self.ms_img_i=tf.image.resize_images(self.ms_img, [self.ms_size, self.ms_size], method=2)

            self.PanSharpening_img,conv4,conv3,conv2,conv1=self.PanSharpening_model_dense_pro(self.pan_img_i, self.ms_img_i)
            self.PanSharpening_img_pan=tf.reshape(tf.reduce_mean(self.PanSharpening_img, axis=3), (batch_size, pan_size, pan_size, 1))
            PanSharpening_img_hp = self.high_pass_2(self.PanSharpening_img_pan)

            self.PanSharpening_img_=tf.image.resize_images(images=self.PanSharpening_img, size=[32, 32],
                                                               method=tf.image.ResizeMethod.BILINEAR)
            self.pan_img_hp=self.high_pass_2(self.pan_img_i, 'pan')

            self.g_spectrum_loss=tf.reduce_mean(tf.square(self.PanSharpening_img_-self.ms_img_i))
            self.g_spatial_loss=tf.reduce_mean(tf.square(PanSharpening_img_hp-self.pan_img_hp))

    def train(self):
        t_vars = tf.trainable_variables()
        d_spatial_vars = [var for var in t_vars if 'spatial_discriminator' in var.name]
        d_spectrum_vars=[var for var in t_vars if 'spectrum_discriminator' in var.name]
        g_vars = [var for var in t_vars if 'Pan_model' in var.name]
        with tf.name_scope('train_step'):
            self.global_step=tf.contrib.framework.get_or_create_global_step()
            self.learning_rate=tf.train.exponential_decay(self.init_lr, global_step=self.global_step, decay_rate=self.lr_decay_rate,
                                                          decay_steps=self.lr_decay_step)
            tf.summary.scalar('global learning rate', self.learning_rate)
            self.train_Pan_model=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.g_loss, var_list=g_vars, global_step=self.global_step)
            self.train_spatial_discrim=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.spatial_loss, var_list=d_spatial_vars)
            self.train_spectrum_discrim=tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.spectrum_loss, var_list=d_spectrum_vars)

    def make_layer(self, scopeName, kernelSize, input, strides=[1, 1, 1, 1], activate=tf.nn.relu, withBN=True, paddingType='SAME'):  #步长为1
        weights = tf.get_variable("%s_w"%scopeName, kernelSize, initializer=tf.truncated_normal_initializer(stddev=1e-3))
        bias = tf.get_variable("%s_b"%scopeName, [kernelSize[-1]], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input, weights, strides=strides, padding=paddingType) + bias
        if(withBN):
            conv1 = tf.contrib.layers.batch_norm(conv1,decay=0.9,updates_collections=None, epsilon=1e-5, scale=True)  #BN层
        output= activate(conv1)
        return output

    def PanSharpening_model_dense_pro(self,pan_img, ms_img):
        with tf.variable_scope('Pan_model'):
            with tf.name_scope('upscale'):
                ms_img=tf.image.resize_images(ms_img, [self.pan_size, self.pan_size], method=2)
            input=tf.concat([ms_img,pan_img],axis=-1)##batch,128,128,5，

            with tf.variable_scope('layer1'):
                conv1_1= self.make_layer('layer1_1',[3, 3, self.num_spectrum+1, 64], input, activate=self.lrelu, withBN=False)
                conv1= self.make_layer('layer1_2',[3, 3, 64, 64], conv1_1, activate=self.lrelu)##batch,128,128,64  #BN

            with tf.variable_scope('layer2'):
                conv2_1= self.make_layer('layer2_1',[3, 3, 64+self.num_spectrum+1, 32], tf.concat([input,conv1],-1), activate=self.lrelu, withBN=False)
                conv2= self.make_layer('layer2_2',[3, 3, 32, 32], conv2_1, activate=self.lrelu)##batch,128,128,32  #BN
            
            with tf.variable_scope('layer3'):
                conv3_1= self.make_layer('layer3_1',[3, 3, 64+self.num_spectrum+1+32, 16], tf.concat([input,conv1,conv2],-1), activate=self.lrelu, withBN=False)
                conv3= self.make_layer('layer3_2',[3, 3, 16, 16], conv3_1, activate=self.lrelu)##batch,128,128,16  #BN
            
            with tf.variable_scope('layer4'):
                conv4_1= self.make_layer('layer4_1',[3, 3, 64+self.num_spectrum+1+32+16, 8], tf.concat([input,conv1,conv2,conv3],-1), activate=self.lrelu, withBN=False)
                conv4= self.make_layer('layer4_2',[3, 3, 8, 8], conv4_1,activate=self.lrelu)##batch,128,128,8   #BN

            with tf.variable_scope('layer5'):
                conv5_1= self.make_layer('layer5_1',[3, 3, 64+self.num_spectrum+1+32+16+8, 4], tf.concat([input,conv1,conv2,conv3,conv4],-1), activate=self.lrelu)
                conv5= self.make_layer('layer5_2',[3, 3, 4, self.num_spectrum], conv5_1, activate=tf.tanh)##batch,128,128,4

        return conv5,conv4,conv3,conv2,conv1

    def spatial_discriminator(self,img_hp,reuse=False,more=None):    # discriminator1
        if(more is None):
            more=[img_hp for i in range(4)]
            chanels=[1,1,1,1]
            flag=1
        else:
            chanels=[64,32,16,8]
            flag=2

        sizes = [[8,8], [16,16], [32,32], [64,64]]
        # sizes = [[64,64],[32,32], [16,16], [8,8]]
        fusionLayer=[]
        ##8,8,1  16,16,1  32,32,1  64,64,1
        with tf.variable_scope('fusion_scope'):
            for i in range(4):
                print(more[i].shape, sizes[i], chanels[i])
                f_=tf.image.resize_images(more[i], sizes[i], method=2)
                fu = self.make_layer('%d_scope_%d'%(flag,i), [1, 1, chanels[i], 1], f_, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                fusionLayer.append(fu)

        with tf.variable_scope('spatial_discriminator', reuse=reuse):
            with tf.variable_scope('layer_1'):
                conv1_spatial_1 = self.make_layer('layer_1_1', [3, 3, 1, 16], img_hp, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv1_spatial = self.make_layer('layer_1_2', [3, 3, 16, 16], conv1_spatial_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv1_spatial=tf.concat([fusionLayer[3],conv1_spatial],axis=-1)  # conv1_spatial=tf.concat([fusionLayer[0],conv1_spatial],axis=-1)
                #64*64*17
            with tf.variable_scope('layer_2'):
                conv2_spatial_1 = self.make_layer('layer_2_1', [3, 3, 17, 32], conv1_spatial, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv2_spatial = self.make_layer('layer_2_2', [3, 3, 32, 32], conv2_spatial_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv2_spatial=tf.concat([fusionLayer[2],conv2_spatial],axis=-1)
                #32*32*33
            with tf.variable_scope('layer_3'):
                conv3_spatial_1 = self.make_layer('layer_3_1', [3, 3, 33, 64], conv2_spatial, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv3_spatial = self.make_layer('layer_3_2', [3, 3, 64, 64], conv3_spatial_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv3_spatial=tf.concat([fusionLayer[1],conv3_spatial],axis=-1)
                # 16*16*65
            with tf.variable_scope('layer_4'):
                conv4_spatial_1 = self.make_layer('layer_4_1', [3, 3, 65, 128], conv3_spatial, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv4_spatial = self.make_layer('layer_4_2', [3, 3, 128, 128], conv4_spatial_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv4_spatial=tf.concat([fusionLayer[0],conv4_spatial],axis=-1)
                # 8*8*129
            with tf.variable_scope('layer_5'):
                conv5_spatial_1 = self.make_layer('layer_5_1', [3, 3, 129, 256], conv4_spatial, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv5_spatial = self.make_layer('layer_5_2', [3, 3, 256, 256], conv5_spatial_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                # 4*4*256
            with tf.variable_scope('line_6'):##same as fc 全连接层
                conv6_spatial = self.make_layer('line_6', [4, 4, 256, 1], conv5_spatial, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=True, paddingType="VALID")
                conv6_spatial=tf.reshape(conv6_spatial, [self.batch_size, 1])
        return conv6_spatial
    

    def spectrum_discriminator(self,img,reuse=False,more=None):     # discriminator2
        if(more is None):
            more=[img for i in range(4)]
            chanels=[self.num_spectrum,self.num_spectrum,self.num_spectrum,self.num_spectrum]
            flag=1
        else:
            chanels=[64,32,16,8]
            flag=2

        sizes = [[8,8], [16,16], [32,32], [64,64]]
        fusionLayer=[]
        ##8,8,1  16,16,1  32,32,1  64,64,1
        with tf.variable_scope('fusion_scope_spectrum'):
            for i in range(4):
                print(more[i].shape, sizes[i], chanels[i])
                f_=tf.image.resize_images(more[i], sizes[i], method=2)
                fu = self.make_layer('%d_scope_%d'%(flag,i), [1, 1, chanels[i], 1], f_, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                fusionLayer.append(fu)

        with tf.variable_scope('spectrum_discriminator', reuse=reuse):
            with tf.variable_scope('layer_1_spectrum'):
                conv1_spectrum_1 = self.make_layer('layer_1_1', [3, 3, self.num_spectrum, 16], img, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv1_spectrum = self.make_layer('layer_1_2', [3, 3, 16, 16], conv1_spectrum_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv1_spectrum=tf.concat([fusionLayer[3],conv1_spectrum],axis=-1)
                #64*64*17
            with tf.variable_scope('layer_2_spectrum'):
                conv2_spectrum_1 = self.make_layer('layer_2_1', [3, 3, 17, 32], conv1_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv2_spectrum = self.make_layer('layer_2_2', [3, 3, 32, 32], conv2_spectrum_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv2_spectrum=tf.concat([fusionLayer[2],conv2_spectrum],axis=-1)
                #32*32*33
            with tf.variable_scope('layer_3_spectrum'):
                conv3_spectrum_1 = self.make_layer('layer_3_1', [3, 3, 33, 64], conv2_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv3_spectrum = self.make_layer('layer_3_2', [3, 3, 64, 64], conv3_spectrum_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv3_spectrum=tf.concat([fusionLayer[1],conv3_spectrum],axis=-1)
                # 16*16*65
            with tf.variable_scope('layer_4_spectrum'):
                conv4_spectrum_1 = self.make_layer('layer_4_1', [3, 3, 65, 128], conv3_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv4_spectrum = self.make_layer('layer_4_2', [3, 3, 128, 128], conv4_spectrum_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                conv4_spectrum=tf.concat([fusionLayer[0],conv4_spectrum],axis=-1)
                # 8*8*129
            with tf.variable_scope('layer_5_spectrum'):
                conv5_spectrum_1 = self.make_layer('layer_5_1', [3, 3, 129, 256], conv4_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=False)
                conv5_spectrum = self.make_layer('layer_5_2', [3, 3, 256, 256], conv5_spectrum_1, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                # 4*4*256
            with tf.variable_scope('line_6_spectrum'):##same as fc  # 相当于全连接层
                conv6_spectrum = self.make_layer('line_6', [4, 4, 256, 1], conv5_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=True, paddingType="VALID")
                conv6_spectrum=tf.reshape(conv6_spectrum, [self.batch_size, 1])
        return conv6_spectrum

    def spectrum_discriminator_1(self,img,reuse=False):
        with tf.variable_scope('spectrum_discriminator', reuse=reuse):
            with tf.variable_scope('layer1_spectrum'):
                conv1_spectrum = self.make_layer('layer1_spectrum', [3, 3, self.num_spectrum, 16], img, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=False)
                # print(conv1_spectrum.shape)
            with tf.variable_scope('layer2_spectrum'):
                conv2_spectrum = self.make_layer('layer2_spectrum', [3, 3, 16, 32], conv1_spectrum, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                # print(conv2_spectrum.shape)
            with tf.variable_scope('layer3_spectrum'):
                conv3_spectrum = self.make_layer('layer3_spectrum', [3, 3, 32, 64], conv2_spectrum, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                # print(conv3_spectrum.shape)
            with tf.variable_scope('layer4_spectrum'):
                conv4_spectrum = self.make_layer('layer4_spectrum', [3, 3, 64, 128], conv3_spectrum, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                #conv4_spectrum = tf.reshape(conv4_spectrum, [self.batch_size, 1 * 1 * 128])
            with tf.variable_scope('layer5_spectrum'):
                conv5_spectrum = self.make_layer('layer5_spectrum', [3, 3, 128, 256], conv4_spectrum, strides=[1, 2, 2, 1], activate=self.lrelu, withBN=True)
                #conv4_spectrum = tf.reshape(conv4_spectrum, [self.batch_size, 1 * 1 * 128])
            with tf.variable_scope('line6_spectrum'):
                conv6_spectrum = self.make_layer('line6_spectrum', [4 , 4 , 256, 1], conv5_spectrum, strides=[1, 1, 1, 1], activate=self.lrelu, withBN=True, paddingType="VALID")
                conv6_spectrum=tf.reshape(conv6_spectrum, [self.batch_size, 1])
                #line5_spectrum = tf.matmul(conv4_spectrum, weights) + bias
                # conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return conv6_spectrum
        
