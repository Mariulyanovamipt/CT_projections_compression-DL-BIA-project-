from comet_ml import Experiment

import argparse
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import os
import CNN_img
import motion
import MC_network
import load
import gc

#for logs in comet ml
exper = Experiment(api_key = "SCNv0oqcoLPJnXIqei3EvhuKR",
                   project_name = "DVC_GAN_1_folder_PSNR_1024_N-256_A_128", #the same forlder as for 1024
                   workspace = 'mariulyanovamipt')

import warnings
warnings.filterwarnings("ignore")

config = tf.ConfigProto(allow_soft_placement=True)
sess = tf.Session(config=config)

parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--l", type=int, default=1024, choices=[256, 512, 1024, 2048])
parser.add_argument("--a", type=int, default=32, choices=[1, 32, 128, 256, 512, 1024, 2048]) #coeff of gan loss
parser.add_argument("--N", type=int, default=256, choices=[128, 256])
parser.add_argument("--M", type=int, default=256, choices=[128, 256])
parser.add_argument("--K", type=int, default=256, choices=[128, 256]) #for discriminator
args = parser.parse_args()

if args.l == 256:
    I_QP = 37
elif args.l == 512:
    I_QP = 32
elif args.l == 1024:
    I_QP = 27
elif args.l == 2048:
    I_QP = 22

batch_size = 4
Height = 736 #256
Width = 64  #256
Channel = 1 #3
lr_init = 1e-4

max_ = 65534
#folder = np.load('folder.npy')
#path = "dataset_video_MY"
#folder = [os.path.join(path, x) for x in os.listdir(path)]
path_one_folder = 'dataset_one_folder_split'
folder_one = os.path.join(path_one_folder, 'folder1')

Y0_com = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel]) #img compressed and restored
Y1_raw = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel]) #img initial
learning_rate = tf.placeholder(tf.float32, [])

with tf.variable_scope("flow_motion"):

    flow_tensor, _, _, _, _, _ = motion.optical_flow(Y0_com, Y1_raw, batch_size, Height, Width)
    # Y1_warp_0 = tf.contrib.image.dense_image_warp(Y0_com, flow_tensor)

# Encode flow
flow_latent = CNN_img.MV_analysis(flow_tensor, args.N, args.M)

entropy_bottleneck_mv = tfc.EntropyBottleneck() #Quantize????
string_mv = entropy_bottleneck_mv.compress(flow_latent)
print(string_mv.shape, 'STRING')
print(string_mv)
print(flow_latent.shape, 'FLOW LATNT')
# string_mv = tf.squeeze(string_mv, axis=0)

flow_latent_hat, MV_likelihoods = entropy_bottleneck_mv(flow_latent, training=True)

print(flow_latent.shape, 'FLOW LATNT hat')

flow_hat = CNN_img.MV_synthesis(flow_latent_hat, args.N)

# Motion Compensation
Y1_warp = tf.contrib.image.dense_image_warp(Y0_com, flow_hat)

MC_input = tf.concat([flow_hat, Y0_com, Y1_warp], axis=-1)
Y1_MC = MC_network.MC(MC_input, filters_final = 1) #?????

# Encode residual
Res = Y1_raw - Y1_MC

res_latent = CNN_img.Res_analysis(Res, num_filters=args.N, M=args.M)

entropy_bottleneck_res = tfc.EntropyBottleneck()
string_res = entropy_bottleneck_res.compress(res_latent)
# string_res = tf.squeeze(string_res, axis=0)

res_latent_hat, Res_likelihoods = entropy_bottleneck_res(res_latent, training=True)

Res_hat = CNN_img.Res_synthesis(res_latent_hat, num_filters=args.N, channels = Channel)

# Reconstructed frame
Y1_com = Res_hat + Y1_MC

'---------------------------------------------DISCRIMINATOR---------------------------------------'
a = args.a
#Y1_com -- fake
#Y1_raw -- real
scores_fake = CNN_img.Discriminator(Y1_com, n_hidden=args.K, n_layer=3)
scores_real = CNN_img.Discriminator(Y1_raw, n_hidden=args.K, n_layer=3)

loss_generator =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_fake, labels=tf.ones_like(scores_real)))
#loss_generator = 0
loss_discriminator = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_real, labels=tf.ones_like(scores_real)) +
    tf.nn.sigmoid_cross_entropy_with_logits(logits=scores_fake, labels=tf.zeros_like(scores_fake)))
loss_adversarial = loss_generator + loss_discriminator
'--------------------------------------------------------------------------------------------------'

# Total number of bits divided by number of pixels.
train_bpp_MV = tf.reduce_sum(tf.log(MV_likelihoods)) / (-np.log(2) * Height * Width * batch_size)
train_bpp_Res = tf.reduce_sum(tf.log(Res_likelihoods)) / (-np.log(2) * Height * Width * batch_size)

# Mean squared error across pixels.
total_mse = tf.reduce_mean(tf.squared_difference(Y1_com, Y1_raw))
warp_mse = tf.reduce_mean(tf.squared_difference(Y1_warp, Y1_raw))
MC_mse = tf.reduce_mean(tf.squared_difference(Y1_raw, Y1_MC))

psnr = 10.0*tf.log(1.0/total_mse)/tf.log(10.0)

# The rate-distortion cost.
l = args.l
a = args.a

train_loss_total = l * total_mse + (train_bpp_MV + train_bpp_Res)
train_loss_total_GAN = l * total_mse + (train_bpp_MV + train_bpp_Res) + a * loss_adversarial
train_loss_total_G = l * total_mse + (train_bpp_MV + train_bpp_Res) + a * loss_generator
train_loss_total_D = l * total_mse + (train_bpp_MV + train_bpp_Res) + a * loss_discriminator
train_loss_MV = l * warp_mse + train_bpp_MV
train_loss_MC = l * MC_mse + train_bpp_MV

# Minimize loss and auxiliary loss, and execute update op.
step = tf.train.create_global_step()

train_MV = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MV, global_step=step)
train_MC = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_MC, global_step=step)
train_total = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total, global_step=step)
train_total_GAN = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total_GAN, global_step=step)
train_total_G = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total_G, global_step=step)
train_total_D = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(train_loss_total_D, global_step=step)

aux_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step = aux_optimizer.minimize(entropy_bottleneck_mv.losses[0])

aux_optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate*10.0)
aux_step2 = aux_optimizer2.minimize(entropy_bottleneck_res.losses[0])

train_op_MV = tf.group(train_MV, aux_step, entropy_bottleneck_mv.updates[0])
train_op_MC = tf.group(train_MC, aux_step, entropy_bottleneck_mv.updates[0])
train_op_all = tf.group(train_total, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])
train_op_all_GAN = tf.group(train_total_GAN, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])
train_op_all_G = tf.group(train_total_G, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])
train_op_all_D = tf.group(train_total_D, aux_step, aux_step2,
                        entropy_bottleneck_mv.updates[0], entropy_bottleneck_res.updates[0])

tf.summary.scalar('mse', total_mse)
tf.summary.scalar('psnr', psnr)
tf.summary.scalar('bits_total', train_bpp_MV + train_bpp_Res)
tf.summary.scalar('adversarial_loss',loss_adversarial)
tf.summary.scalar('generator_loss',loss_generator)
tf.summary.scalar('discriminator_loss',loss_discriminator)
#save_path = './OpenDVC_PSNR_' + str(l) + 'one_folder'#+'_minmax_norm_N_M_128'
save_path = './OpenDVC_PSNR_'+'GAN'+str(l)+str(a)+'_N_'+str(args.N)
checkpoint_path = os.path.join(save_path, 'model.ckpt')
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
saver = tf.train.Saver(max_to_keep=None)

'''
sess.run(tf.global_variables_initializer())
var_motion = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
print('TRAINABLE VARS', tf.GraphKeys.TRAINABLE_VARIABLES)
print('VAR MOTION', var_motion )
print(saver)
saver_motion = tf.train.Saver(var_list=var_motion, max_to_keep=None)
saver_motion.save(sess, checkpoint_path)# , global_step  = tf.Variable(0, name='global_step', trainable=False))
print('_____________', saver_motion)
print('SESS', sess)
saver_motion.restore(sess, save_path='motion_flow/model.ckpt-0')
'''
sess.run(tf.global_variables_initializer())
'''
var_motion = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='flow_motion')
saver_motion = tf.train.Saver(var_list=var_motion, max_to_keep=None)
saver_motion.restore(sess, save_path='OpenDVC_PSNR_1024one_folder/model.ckpt-1080')
'''

saver.restore(sess, save_path=save_path+'/model.ckpt-600')

#pretrained motion models?? /from readme


#saver_psnr = tf.train.Saver(max_to_keep=None)

##latest = tf.train.latest_checkpoint(checkpoint_dir='./OpenDVC_PSNR_' + str(l * 32))
##latest = tf.train.latest_checkpoint(checkpoint_dir='./OpenDVC_MS-SSIM_' + str(l * 32))
#saver_psnr.restore(sess, save_path=save_path)

# Train
iter = 599

while(True):

    #if iter <= 100000:
    if iter <= 500:
        #frames = 2
        frames = 10

        #if iter <= 20000:
        if iter <= 300:
            train_op = train_op_MV
        #elif iter <= 40000:
        elif iter <= 600: #1400
            train_op = train_op_MC
        elif iter <= 900: #2603
            train_op = train_op_all
        elif (iter%2 == 0):
            train_op = train_op_all_G
        else:
            train_op = train_op_all_D
    else:
        #frames = 7
        frames = 30
        train_op = train_op_all

    #if iter <= 300000:
    if iter <= 3000:
        lr = lr_init
    #elif iter <= 600000:
    elif iter <= 5000:
        lr = lr_init / 10.0
    else:
        lr = lr_init / 100.0

    data = np.zeros([frames, batch_size, Height, Width, Channel])
    #data = load.load_data_ssim_pydicom(data, folder, frames, batch_size, Height, Width, Channel)
    data = load.load_pydicom_one_folder(data, folder_one, frames, batch_size, Height, Width, Channel)

    total_mse_train = 0

    for ff in range(frames-1):

        if ff == 0:

            F0_com = data[0]
            F1_raw = data[1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com/ max_,
                                                Y1_raw: F1_raw/ max_,
                                                learning_rate: lr})

        else:

            F0_com = F1_decoded * max_
            F1_raw = data[ff+1]

            _, F1_decoded = sess.run([train_op, Y1_com],
                                     feed_dict={Y0_com: F0_com/max_,
                                                Y1_raw: F1_raw/max_,
                                                learning_rate: lr})

        print('Training_OpenDVC Iteration:', iter)

        iter = iter + 1
        total_mse_train += tf.reduce_mean(tf.squared_difference(F1_decoded, F1_raw))/frames
        #loss_summ = total_mse_train.eval(session=tf.compat.v1.Session())
        loss_summ = np.add(total_mse_train, 0)

        if iter % 25 == 0:
        #if iter % 500 == 0:

             merged_summary_op = tf.summary.merge_all()
             summary_str = sess.run(merged_summary_op, feed_dict={Y0_com: F0_com/max_,
                                                                  Y1_raw: F1_raw/max_})

             summary_writer.add_summary(summary_str, iter)
             exper.log_metric('Loss_train', loss_summ, step = iter)
             #summary_writer.add_summary(summary=loss_summ, global_step=iter)

        if iter % 50 == 0:
        #if iter % 20000 == 0:

             checkpoint_path = os.path.join(save_path, 'model.ckpt')
             saver.save(sess, checkpoint_path, global_step=iter)

    if iter > 3500:
        break

    del data
    del F0_com
    del F1_raw
    del F1_decoded

    gc.collect()
