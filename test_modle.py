'''
测试模型，使用到训练好的权重文件checkpoint/g_semi.npz
只用到半监督分割模型分割所选的图像，需要关闭弹出来的提示框'系统正在分割，请稍等...... '
'''
from config import config as conf
from utils import *
from model import *
import time
import os
import tkinter as tk
from save_result import *
# 传入参数实例： test_image_file = 'test_img/img/58.PNG'
def test_modle(test_image_file, threshod_new):
    # 读入真实的分割结果seg，如果要在一张图像上显示结果，必须修改成测试图像真实分割结果的路径
    seg_truth_path = 'data/seg/58.PNG'
    seg = cv2.imread(seg_truth_path, cv2.IMREAD_GRAYSCALE)
    total_result_path = 'result/total_result.png'
    seg_result_path = 'result/seg_result.png'
    # # 显示提示信息
    # messagebox.showinfo(title='上传成功', message='系统正在分割，请稍等...... ')
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # create folders to save result images and trained model
    checkpoint_dir = conf.TRAIN.ckpt_dir
    tl.files.exists_or_mkdir(checkpoint_dir)

    # load data
    test_img = np.array(np.reshape(tl.vis.read_image(test_image_file), (1, 256, 256, 1)))/127.5 - 1.
    test_seg = np.zeros(test_img.shape,dtype=test_img.dtype)

    # define model
    x_test = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    y_test = tf.placeholder(tf.float32, shape=[None, 256, 256, 1])
    # 模型初始化
    gm_tanh, gm_logit = GAN_g(x_test, n_classes=1, is_train=False, dropout=1.)

    # test
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # load model param
        tl.files.load_and_assign_npz(sess=sess, name=conf.TRAIN.g_model, network=gm_logit)

        # 此处可以对测试图像test_img进行数据增强预处理
        feed_dict = {x_test: test_img,y_test : test_seg}

        t_start = time.time() # 开始测试
        _out = sess.run(gm_logit.outputs, feed_dict=feed_dict)

        # print('模型的输出：  ',_out)
        # print(_out.shape) # (1, 256, 256, 1)

        # 对输出进行降维后输出
        img = _out.squeeze()
        # print('用于输出的数组：  ',img)
        # print(img.shape)
        # print(type(img))
        # print('最小像素值：  ',img.min())
        # print('最大像素值：  ',img.max())
        # print('平均像素值：  ', img.mean())
        # cv2.imwrite(seg_result_path, img)  # 保存模型的输出结果,由于没有二值化处理，得到的是全黑的图像
        # 给全部的像素点按照阈值划分前景和背景
        threshod = threshod_new
        for i in range(256):
            for j in range(256):
                if img[i, j] >= float(threshod):
                    img[i, j] = 255.
                else:
                    img[i, j] = 0.

        cv2.imwrite(seg_result_path , img)  # 保存模型的输出结果

    print("评价指标：")
    # 计算IoU
    IoU = eval_IoU(img, seg, axis=(0, 1))
    print('IoU = ', '%.3f' % IoU)
    # '%.2f' % f
    dice = eval_dice_hard(img, seg,axis=(0,1))
    print('Dice = ', '%.3f' % dice)

    # _, precision, recall, _ = eval_tfpn(_out > 0, seg_feed)
    acc, precision, recall, TNR = eval_tfpn(img, seg,axis=(0,1))
    print('accruate = ', '%.3f' % acc)
    # print('precision = ', precision)
    print('recall = ', '%.3f' % recall)
    # print('TNR = ', TNR)
    print('分割用时： ', '%.3f' % (time.time() - t_start))


    # 可视化分割结果：将原图和分割结果放在一张图片上展示
    imgfile = test_image_file #'data/image/58.PNG'
    maskfile = seg_result_path  # 系统分割结果的路径'result/img.png'
    maskfile_truth = seg_truth_path # 真实标签的路径'data/seg/58.PNG'

    img = cv2.imread(imgfile, 1)
    mask = cv2.imread(maskfile, 0) # 系统分割结果展示
    mask_truth = cv2.imread(maskfile_truth, 0) # 真实标签，测地真值展示

    contours1, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask_truth, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours1, -1, (255, 0, 0), 1)
    cv2.drawContours(img, contours2, -1, (0, 255, 0), 1) # 加上测地真值的轮廓，绿色
    img = img[:, :, ::-1]
    img[..., 2] = np.where(mask == 1, 255, img[..., 2])

    cv2.imwrite(total_result_path,img)
    # 至此保存分割结果完毕


# 定义新的窗口用来展示分割结果，并保存医生病例描述信息
    # 第1步，实例化object，建立窗口window
    window = tk.Toplevel()

    # 第2步，给窗口的可视化起名字
    window.title('图像分割结果展示及病例信息填写')

    # 第3步，设定窗口的大小(长 * 宽)
    window.geometry('600x600')  # 这里的乘是小x

    # 读取图片
    logo = tk.PhotoImage(file = total_result_path)
    # label展示图片
    label_image = tk.Label(window, image = logo).place(x=170,y=50)
    # label展示文字
    label_text = tk.Label(window,justify='left',padx=10,text="""医生您好！图示为系统的图像分割结果:""").place(x=170,y=20)

    # 第4步，在图形界面上设定标签
    # 病例名称
    new_name = tk.StringVar()
    new_name.set('58.PNG')
    tk.Label(window, text='病例名称:').place(x=170, y=350)
    tk.Entry(window, textvariable=new_name).place(x=250, y=350)

    # 网络名称
    new_modle = tk.StringVar()
    new_modle.set('TUS-GAN')
    tk.Label(window, text="网络名称:").place(x=170, y=389)
    tk.Entry(window, textvariable=new_modle).place(x=250, y=389)

    # 病例分析
    new_describe = tk.StringVar()
    new_describe.set('经初步观察，此结节为')
    tk.Label(window, text="病例分析:").place(x=170, y=430)
    tk.Entry(window, textvariable=new_describe).place(x=250, y=430)

    # 加入保存的按钮，调用save_result函数
    b = tk.Button(window, text='保存病例信息', command=lambda :save_result(window,img,new_name.get(),new_modle.get(),new_describe.get())).place(x=240,y=500)

    # 第6步，主窗口循环显示
    window.mainloop()


if __name__ == '__main__':
    test_modle(test_image_file = 'data/image/58.PNG',threshod_new=3)

