# -*- coding: utf8 -*-
# 定义训练网络函数train_gan()和验证网络函数evaluate()
# 在main()函数中传入参数，控制具体调用的函数
# 运行main.py训练与验证网络，将权重文件保存在checkpoint/g_semi.npz内
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
import time
from tqdm import trange

from model import GAN_g, GAN_d
from config import config as conf
from loss import infer_g_init_train, infer_g_valid, seg_loss
from utils import load_csv_list, crop_sub_imgs_fn, load_and_assign_npz, eval_H_dist, eval_tfpn, eval_dice_hard, eval_IoU

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 在这里指定GPU的ID号

def train_gan():
    # create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir # 'checkpoint/' 将参数从config中导入到main.py
    tl.files.exists_or_mkdir(checkpoint_dir) # 构造用于储存图片的文件夹，同时定义checkpoint的文件夹
    samples_dir = conf.TRAIN.gan_samples_dir # 'samples/gan' 将参数从config中导入到main.py
    tl.files.exists_or_mkdir(samples_dir)
    logs_dir = conf.TRAIN.gan_log # 'logs/log_gan' 将参数从config中导入到main.py
    tl.files.exists_or_mkdir(logs_dir)

    # Adam
    lr_init = conf.TRAIN.lr_init * 0.1 # 初始学习率 = 1e-2 所以lr_init = 0.001
    beta1 = conf.TRAIN.beta1 # 0.9
    batch_size = conf.TRAIN.batch_size # 10
    ni = int(np.ceil(np.sqrt(batch_size)))

    # load data 载入数据
    train_img_list, _ = load_csv_list(conf.TRAIN.img_list_path)
    train_img_list2, _ = load_csv_list(conf.TRAIN.img_list_path2)
    valid_img_list, _ = load_csv_list(conf.VALID.img_list_path)

    train_imgs = (np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.img_path, n_threads=32)), axis=3))/127.5 - 1.
    train_segs = np.expand_dims(np.array(tl.vis.read_images(train_img_list, path=conf.TRAIN.seg_path, n_threads=32))>0.5, axis=3)
    train_imgs2 = (np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.img_path, n_threads=32)), axis=3))/127.5 - 1.
    train_segs2 = np.expand_dims(np.array(tl.vis.read_images(train_img_list2, path=conf.TRAIN.seg_path, n_threads=32))>0.5, axis=3)
    valid_imgs = (np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=32)), axis=3))/127.5 - 1.
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=32))>0.5, axis=3)
    train_data = np.concatenate((train_imgs, train_segs), axis=3)

    # vis data
    vidx = 0
    train_vis_img = train_imgs[vidx:vidx+batch_size,:,:,:]
    train_vis_seg = train_segs[vidx:vidx+batch_size,:,:,:]
    valid_vis_img = valid_imgs[vidx:vidx+batch_size,:,:,:]
    valid_vis_seg = valid_segs[vidx:vidx+batch_size,:,:,:]

    # 保存图片
    tl.vis.save_images(train_vis_img, [ni,ni], os.path.join(samples_dir, '_train_img.png'))
    tl.vis.save_images(train_vis_seg, [ni,ni], os.path.join(samples_dir, '_train_seg.png'))
    tl.vis.save_images(valid_vis_img, [ni,ni], os.path.join(samples_dir, '_valid_img.png'))
    tl.vis.save_images(valid_vis_seg, [ni,ni], os.path.join(samples_dir, '_valid_seg.png'))

    # define network
    x_m = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    y_m = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    x_n = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    y_n = tf.placeholder(tf.float32, shape=[None, 224, 224, 1])
    x_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

# 构造生成网络，获得生成网络的输出gm_logit，生成网络的本质是DenseDeepLab_1
    gm_tanh, gm_logit = GAN_g(x_m, n_classes=1, is_train=True)# Seg(X_a)训练标注数据，生成标注数据的分割网络
    gn_tanh, _ = GAN_g(x_n, n_classes=1, is_train=True, reuse=True)# Seg(X_u)训练未标注数据，生成未标注数据的分割网络
    g_dice = tl.cost.dice_hard_coe(gm_logit.outputs, y_m, threshold=0, axis=[1,2,3])# Dice(Seg(X_a),Y_a)计算标注数据和ground_truth的Dice损失
    v_g, v_dice = infer_g_valid(x_valid, y_valid)
    #g_vars = g_logit.all_params

# 构造判别网络，比较生成的输出gm_tanh和目标图像，d_logit1_fake表示整个网络
    d_logit1_real = GAN_d(x_m, y_m, is_train=True)# Eva(X_a,Y_a)
    d_logit1_fake0 = GAN_d(x_m, gm_tanh, is_train=True, reuse=True)# Eva(X_a,Seg(X_a))
    d_logit1_fake = GAN_d(x_n, gn_tanh, is_train=True, reuse=True)# Eva(X_u,Seg(X_u))
    #d_vars = d_logit1_real.all_params


# 损失函数中的超参数
    lambda_adv = 0.02
    lambda_a = 0.5
    lambda_u = 1 - lambda_a

    # 定义损失函数
    d_l1_loss1 = tl.cost.sigmoid_cross_entropy(d_logit1_real.outputs, tf.ones_like(d_logit1_real.outputs), name='d_l1_1')
    d_l1_loss2 = lambda_a * tl.cost.sigmoid_cross_entropy(d_logit1_fake0.outputs, tf.zeros_like(d_logit1_fake0.outputs), name='d_l1_2')
    d_l1_loss3 = lambda_u * tl.cost.sigmoid_cross_entropy(d_logit1_fake.outputs, tf.zeros_like(d_logit1_fake.outputs), name='d_l1_3')
# 判别网络部分的总损失L_Eva = l_bce + lambda_a * l_bce_a + lambda_u * l_bce_u
    d_loss = d_l1_loss1 + d_l1_loss2 + d_l1_loss3

    g_seg_loss = seg_loss(gm_logit.outputs, y_m)# L_seg = l_dce(Seg(X_a),Y_a) + l_bce(Seg(X_a),Y_a)
    g_gan_loss1 = lambda_adv * lambda_a * tl.cost.sigmoid_cross_entropy(d_logit1_fake0.outputs, tf.ones_like(d_logit1_fake0.outputs), name='g_gan1')
    g_gan_loss2 = lambda_adv * lambda_u * tl.cost.sigmoid_cross_entropy(d_logit1_fake.outputs, tf.ones_like(d_logit1_fake.outputs), name='g_gan2')
# 分割网络部分的总损失L_S = L_Seg + lambda_adv * L_adv = L_Seg + lambda_adv * lambda_a * l_bce_a + lambda_adv * lambda_u * l_bce_u
    g_loss = g_seg_loss + g_gan_loss1 + g_gan_loss2

    # vars
# 获取参数列表,常常用于选择哪些参数需要被更新，比如训练 GAN 时，可以分别获取 G 和 D 的参数列表，放到对应的 optimizer 中。
    g_vars = tl.layers.get_variables_with_name('DenseDeepLab', True, True)
    d_vars = tl.layers.get_variables_with_name('DenseAttenNet', True, True)

    #Train Operation
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    
    g_var1 = [v for v in g_vars if 'ASPP' in v.name]
    g_var2 = [v for v in g_vars if 'ASPP' not in v.name]
    
    ## Pretrain
# 生成网络使用Adam优化器
    g_optim_1 = tf.train.AdamOptimizer(lr_v*10, beta1=beta1).minimize(g_loss, var_list=g_var1)
    g_optim_2 = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_var2)
    g_optim = tf.group(g_optim_1, g_optim_2)

# 判别网络使用梯度下降优化
    d_optim = tf.train.GradientDescentOptimizer(lr_v*5).minimize(d_loss, var_list=d_vars)

    # train训练
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        #sess.run(tf.global_variables_initializer())
        tl.layers.initialize_global_variables(sess)
        
        # summary
        # 使用命令查看训练过程图像： tensorboard --logdir="logs/log_gan"
        # http://localhost:6006/
        tb_writer = tf.summary.FileWriter(logs_dir, sess.graph) # 'logs/log_gan'
        tf.summary.scalar('loss_d/loss_d', d_loss)
        tf.summary.scalar('loss_d/loss_d_l1r', d_l1_loss1)
        tf.summary.scalar('loss_d/loss_d_l1f', d_l1_loss2)
        tf.summary.scalar('loss_d/loss_d_l1f0', d_l1_loss3)
        #tf.summary.scalar('loss_d/loss_d_l2r', d_l2_loss1)
        #tf.summary.scalar('loss_d/loss_d_l2f', d_l2_loss2)
        tf.summary.scalar('loss_g/loss_g', g_loss)
        tf.summary.scalar('loss_g/loss_gan1', g_gan_loss1)
        tf.summary.scalar('loss_g/loss_gan2', g_gan_loss2)
        tf.summary.scalar('loss_g/loss_seg', g_seg_loss)
        tf.summary.scalar('dice', g_dice)
        tf.summary.scalar('learning_rate', lr_v)
        tb_merge = tf.summary.merge_all()

        # load model
        # load_and_assign_npz(sess=sess, model_path=checkpoint_dir, model_name=conf.TRAIN.g_model.split('/')[-1], var_list=g_vars)
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.g_model, network=gm_logit) # 'checkpoint/g_semi.npz'
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.d_model1, network=d_logit1_real) # 'checkpoint/d1_semi.npz'
        # load_and_assign_npz(sess=sess, model_path=checkpoint_dir, model_name=conf.TRAIN.d_model.split('/')[-1], var_list=d_vars)
        print("successfully load npz!")
        # datasets information
        n_epoch = conf.TRAIN.n_epoch # 5000 实际用的是10
        lr_decay = conf.TRAIN.lr_decay # 0.2
        decay_every = conf.TRAIN.decay_every # int(config.TRAIN.n_epoch / 5)
        n_step_epoch = np.int(len(train_imgs)/batch_size) # 一个epoch里有n_step_epoch个batch
        n_step = n_epoch * n_step_epoch
        # val_step_epoch = np.int(val_fX.shape[0]/FLAGS.batch_size)
    
        print('\nInput Data Info:')
        print('   train_file_num:', len(train_imgs), '\tval_file_num:', len(valid_imgs))
        print('\nTrain Params Info:')
        print('   learning_rate:', lr_init)
        print('   batch_size:', batch_size)
        print('   n_epoch:', n_epoch, '\tstep in an epoch:', n_step_epoch, '\ttotal n_step:', n_step)
        print('\nBegin Training ...')
    
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        max_dice = 0
        tb_train_idx = 0
        for epoch in range(n_epoch):
            # update learning rate
            if epoch != 0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay**(epoch // decay_every) # 整数除//、幂运算**
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
                print(log)

            # time_start = time.time()
            t_batch = [x for x in tl.iterate.minibatches(inputs=train_data, targets=train_data, batch_size=batch_size, shuffle=True)]
            t_batch2 = [x for x in tl.iterate.minibatches(inputs=train_imgs2, targets=train_segs2, batch_size=batch_size, shuffle=True)]

        # 加上进度条，trange(i) 是 tqdm(range(i)) 的另一种写法
            tbar = trange(min(len(t_batch), len(t_batch2)), unit='batch', ncols=100)
            train_err_d, train_err_g, train_dice, n_batch = 0, 0, 0, 0
            for i in tbar:
                # You can also use placeholder to feed_dict in data after using
                # img_seg = np.concatenate((batch[i][0], batch[i][1]), axis=3)

                # 对每个batch里的图像数据进行增强，保存到投喂字典feed_dict中
                img_seg = tl.prepro.threading_data(t_batch[i][0], fn=crop_sub_imgs_fn, is_random=True)
                img_feed = np.expand_dims(img_seg[:,:,:,0], axis=3)
                seg_feed = np.expand_dims(img_seg[:,:,:,1], axis=3)
                xn_img_feed = tl.prepro.threading_data(t_batch2[i][0], fn=crop_sub_imgs_fn, is_random=False)
                yn_img_feed = tl.prepro.threading_data(t_batch2[i][1], fn=crop_sub_imgs_fn, is_random=False)
                feed_dict = {x_m: img_feed, y_m: seg_feed, x_n: xn_img_feed, y_n: yn_img_feed}

                # update D
                # sess.run(d_optim, feed_dict=feed_dict)
                _errD, _errDl11, _errDl12, _errDl13, _ = sess.run([d_loss, d_l1_loss1, d_l1_loss2, d_l1_loss3, d_optim], feed_dict=feed_dict)

                # update G
                _tbres, _dice, _errG, _errSeg, _errGAN1, _errGAN2, _ = sess.run([tb_merge, g_dice, g_loss, g_seg_loss, g_gan_loss1, g_gan_loss2, g_optim], feed_dict=feed_dict)

                train_err_g += _errG; train_err_d += _errD; train_dice += _dice; n_batch += 1

                tbar.set_description('Epoch %d/%d ### step %i' % (epoch+1, n_epoch, i)) # epoch里是第i步(step)，一个step里一个batch
            # 设置后缀
                tbar.set_postfix(dice=train_dice/n_batch, g=train_err_g/n_batch, d=train_err_d/n_batch, g_seg=_errSeg, g_gan=_errGAN1+_errGAN2, d_11=_errDl11, d_12=_errDl12+_errDl13)

                tb_writer.add_summary(_tbres, tb_train_idx)
                tb_train_idx += 1
            
            if np.mod(epoch, conf.VALID.per_print) == 0: # per_print = 5
                # vis image
                feed_dict = {x_valid: train_vis_img, y_valid: train_vis_seg}
                feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g.outputs, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'train_pred_{}.png'.format(epoch)))

                feed_dict = {x_valid: valid_vis_img, y_valid: valid_vis_seg}
                feed_dict.update(v_g.all_drop)
                _output = sess.run(v_g.outputs, feed_dict=feed_dict)
                tl.vis.save_images(_output, [ni,ni], os.path.join(samples_dir, 'valid_pred_{}.png'.format(epoch)))

                print('Validation ...')
                time_start = time.time()
                val_acc, n_batch = 0, 0
                for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=True):
                    img_feed, seg_feed = batch
                    feed_dict = {x_valid: img_feed, y_valid: seg_feed}
                    feed_dict.update(v_g.all_drop)
                    _dice = sess.run(v_dice, feed_dict=feed_dict)
                    val_acc += _dice; n_batch += 1
                print('   Time:{}\tDice:{}'.format(time.time()-time_start, val_acc/n_batch))

                if val_acc/n_batch > max_dice:
                    max_dice = val_acc/n_batch
                print('[!] Max dice:', max_dice)

        tl.files.save_npz(gm_logit.all_params, name=conf.TRAIN.g_model) # 'checkpoint/g_semi.npz'
        tl.files.save_npz(d_logit1_real.all_params, name=conf.TRAIN.d_model1) # 'checkpoint/d1_semi.npz'

def evaluate():
    # create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir # 'checkpoint/'
    tl.files.exists_or_mkdir(checkpoint_dir)
    result_dir = conf.VALID.result_dir # 'valid_result/'
    tl.files.exists_or_mkdir(result_dir)

    # load data
    valid_img_list, _ = load_csv_list(conf.VALID.img_list_path) # 'data/list/list.csv'
    valid_imgs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.img_path, n_threads=32)), axis=3) / 127.5 - 1.
    valid_segs = np.expand_dims(np.array(tl.vis.read_images(valid_img_list, path=conf.VALID.seg_path, n_threads=32))>0.5, axis=3)
    
    # define model
    # tf.compat.v1.disable_eager_execution()
    x_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_valid = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])

    gm_tanh, gm_logit = GAN_g(x_valid, n_classes=1, is_train=False, dropout=1.)
    oris = []
    segs = []
    pred_maps = []

    # valid
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # sess.run(tf.global_variables_initializer())
        # tl.layers.initialize_global_variables(sess)

        # load model
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.g_model, network=gm_logit)

        # 定义结果字典results,用于存储各个评价指标
        results = {'dice': [], 'IoU':[],'precision':[],'recall':[],'FScore': [], 'HDist': [], 'AvgDist':[],'Obj': [], 'Eva_Time': []}

        for batch in tl.iterate.minibatches(inputs=valid_imgs, targets=valid_segs, batch_size=1, shuffle=False):
            img_feed, seg_feed = batch
            feed_dict = {x_valid: img_feed, y_valid: seg_feed}
            print('原始图像：  ',img_feed)
            print('分割图像：  ',seg_feed)

            t_start = time.time()# 开始验证
            _out = sess.run(gm_logit.outputs, feed_dict=feed_dict)
            print('输出：   ',_out)
            pred_maps.append(_out)
            oris.append(img_feed)
            segs.append(seg_feed)

# 计算各个评价指标dice, precision,recall, HD,AvgDist,IoU
            _, precision, recall, _ = eval_tfpn(_out>0, seg_feed)
            HD, AD = eval_H_dist(_out>0, seg_feed)
            IoU = np.mean(eval_IoU(_out>0, seg_feed))

            results['dice'].append(np.mean(eval_dice_hard(_out>0, seg_feed)))
            results['IoU'].append(IoU)
            results['precision'].append(np.mean(precision))
            results['recall'].append(np.mean(recall))
            results['HDist'].append(HD)
            results['AvgDist'].append(AD)
            results['Obj'].append(0 if np.sum(_out>0) == 0 else 1)
            results['Eva_Time'].append(time.time()-t_start)

        # 把原始图像、分割结果和预测图像保存到 结果路径result_dir = 'valid_result/'
        np.save(os.path.join(conf.VALID.result_dir, 'pred.npy'), np.array(pred_maps))
        np.save(os.path.join(conf.VALID.result_dir, 'ori.npy'), np.array(oris))
        np.save(os.path.join(conf.VALID.result_dir, 'seg.npy'), np.array(segs))

# 把评价指标结果 保存到result.csv中
        with open(os.path.join(conf.VALID.result_dir, 'results.csv'), 'w') as f:
            f.write('name,dice,IoU,precision,recall,HDist,AvgDist,Obj,Time\n')
            for i, valid_name in enumerate(valid_img_list):
                f.write('{},{},{},{},{},{},{},{},{}\n'.format(valid_name, results['dice'][i], results['IoU'][i], results['precision'][i], results['recall'][i], results['HDist'][i], results['AvgDist'][i], results['Obj'][i], results['Eva_Time'][i]))
        print("验证数据集的长度：",len(results['dice']))
        print('平均dice:',np.mean(results['dice']),'\t标准差：', np.std(results['dice']))# numpy.std(arr，axis = None)：计算指定数据(数组元素)沿指定轴(如果有)的标准偏差。
        print('平均HDist:',np.mean(results['HDist']),'\t标准差：', np.std(results['HDist']))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train_gan', help='train_gan, evaluate')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'train_gan':
        train_gan()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
