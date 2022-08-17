import tensorflow as tf
import os
import time
from PanGan_pro import PanGan
from DataSet import DataSet
from config import FLAGES

SPATI_SPECT = 0
#SPATI_SPECT=1
#SPATI_SPECT=2
###
def print_current_training_stats(error_pan_model, error_ms_model, error_g_model, global_step, learning_rate, time_elapsed):
    stats = 'Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial loss: {}'.format(error_pan_model)
    losses += ' | spectrual loss: {}'.format(error_ms_model)
    losses += ' | generator loss: {}'.format(error_g_model)
    print(stats)
    print(losses + '\n')
    
def print_current_training_stats_valid(error_spatial, error_spectrual, global_step, learning_rate, status, time_elapsed):
    Mse_p,Rmse_p,Psnr_p, Mse_m,Rmse_m,Psnr_m = status
    stats = 'Valid_Step: {}/{} ----- Cur_lr: {:1.7f} ----- Time: {:>2.2f} sec.'.format(global_step, FLAGES.iters,
                                                                                 learning_rate, time_elapsed)
    losses =  ' | spatial error: {}'.format(error_spatial)
    losses += ' | spectrual error: {}'.format(error_spectrual)
    losses += ' | Mse_p: {},Rmse_p: {},Psnr_p: {}, Mse_m: {},Rmse_m: {},Psnr_m: {}'.format(Mse_p,Rmse_p,Psnr_p, Mse_m,Rmse_m,Psnr_m)
    print(stats)
    print(losses + '\n')

def main(argv):
    model=PanGan(FLAGES.pan_size, FLAGES.ms_size, FLAGES.batch_size, FLAGES.num_spectrum, FLAGES.ratio,
                 FLAGES.lr,FLAGES.decay_rate,FLAGES.decay_step,is_training=True)
    model.train()
    print("-----------Prepare train dataset: , num_spectrum: %d---------------------"%FLAGES.num_spectrum)
    dataset=DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                    FLAGES.stride, FLAGES.num_spectrum, FLAGES.dataType)
    DataGenerator_train=dataset.generator_train()
    DataGenerator_valid=dataset.generator_valid()
    print("-----------Finish build dataset----------------------")

    merge_summary=tf.summary.merge_all()
    if not os.path.exists(FLAGES.log_dir):
        os.makedirs(FLAGES.log_dir)
    if not os.path.exists(FLAGES.model_save_dir):
        os.makedirs(FLAGES.model_save_dir)

    with tf.Session() as sess:
        train_writer=tf.summary.FileWriter(FLAGES.log_dir, sess.graph)
        saver=tf.train.Saver(max_to_keep=None)
        saver_g=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'Pan_model'),max_to_keep=None)
        init_op=tf.global_variables_initializer()
        sess.run(init_op)

        if FLAGES.is_pretrained:
            print("Restore pretrained model from: %s"%FLAGES.pretrained_model)
            saver.restore(sess, FLAGES.pretrained_model)
        print("\n------------------------Now start to train----------------------------")
        for training_itr in range(FLAGES.iters):
            t1 = time.time()
            pan_batch, ms_batch=next(DataGenerator_train)
            for i in range(2):
                if(SPATI_SPECT==0 or SPATI_SPECT==1): 
                    _, error_pan_model = sess.run([model.train_spatial_discrim, model.spatial_loss],feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
                if(SPATI_SPECT==0 or SPATI_SPECT==2): 
                    _, error_ms_model = sess.run([model.train_spectrum_discrim, model.spectrum_loss],feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch })

            if(SPATI_SPECT==0):
                _, error_g_model, global_step, summary, learning_rate = sess.run([model.train_Pan_model, model.g_loss,model.global_step, merge_summary,model.learning_rate],
                                                                             feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            if(SPATI_SPECT==1):
                _, error_g_model, global_step, summary, learning_rate = sess.run([model.train_Pan_model, model.g_loss1,model.global_step, merge_summary,model.learning_rate],
                                                                             feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            if(SPATI_SPECT==2):
                _, error_g_model, global_step, summary, learning_rate = sess.run([model.train_Pan_model, model.g_loss2,model.global_step, merge_summary,model.learning_rate],
                                                                             feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            error_pan_model=sess.run(model.spatial_loss,feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})
            error_ms_model=sess.run(model.spectrum_loss,feed_dict={model.pan_img: pan_batch, model.ms_img: ms_batch})					
            print_current_training_stats(error_pan_model, error_ms_model, error_g_model, global_step, learning_rate, time.time()-t1)
            train_writer.add_summary(summary, global_step)
            
            if (global_step + 1) %  FLAGES.valid_iters == 0:
                t1 = time.time()
                print("----------------Valid model-----------------------")
                pan_valid_batch, ms_valid_batch=next(DataGenerator_valid)
                error_spatial,error_spectrum, Mse_p,Rmse_p,Psnr_p, Mse_m,Rmse_m,Psnr_m = \
                    sess.run([model.valid_spatital_error,model.valid_spectrum_error, model.Mse_p,model.Rmse_p,model.Psnr_p, model.Mse_m,model.Rmse_m,model.Psnr_m], \
                    feed_dict={model.pan_img: pan_valid_batch, model.ms_img: ms_valid_batch})
                status = [Mse_p,Rmse_p,Psnr_p, Mse_m,Rmse_m,Psnr_m]
                print_current_training_stats_valid(error_spatial, error_spectrum, global_step, learning_rate, status, time.time()-t1)
            
            if (global_step + 1) %  FLAGES.model_save_iters == 0:
                saver.save(sess=sess, save_path=FLAGES.model_save_dir + '/' + 'PanNet', global_step=(global_step+1) )
                saver_g.save(sess=sess, save_path=FLAGES.model_save_dir + '/' + 'Generator', global_step=(global_step+1)  )
                print('\nModel checkpoint saved...\n')

            if global_step == FLAGES.iters:
                break
        print('Training done.')

if __name__ == '__main__':
    tf.app.run()



