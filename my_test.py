import numpy as np
datas = np.load('checkpoint/g_semi.npz')
datas_file = datas.files
# print(datas['params'])
print("生成网络-分割网络的参数： ",datas_file)

datas2 = np.load('checkpoint/d1_semi.npz')
datas_file2 = datas2.files
# print(datas2['params'])
print("判别网络-评判网络的参数： ",datas_file2)


test = np.load('./valid_result/pred.npy',encoding='latin1')
doc = open('./valid_result/ori.txt','a')
print(test,file=doc)
print(test)