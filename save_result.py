'''
选择保存病例信息的文件夹，并将分割后的图片total_result.png和病例信息写入result_information.txt文件
'''
from tkinter import filedialog,messagebox
import cv2

def save_result(root, img, new_name,new_model,new_describe):
    save_path = "./result"
    File = filedialog.askdirectory(parent=root, initialdir=save_path, title='选择保存路径')
    cv2.imwrite(f'{File}/total_result.png', img)
    f = open(f'{File}/result_information.txt', "w")
    f.write('病例信息：')
    f.write('\r\n')
    f.write(new_name)
    f.write('\r\n')
    f.write('模型名称：')
    f.write(new_model)
    f.write('\r\n')
    f.write(new_describe)
    f.write('\r\n')
    f.write('注：红色为模型的分割结果。')
    f.close()
    # 显示提示信息
    messagebox.showinfo(title='该病人的病例信息保存成功',message='保存路径为: ' + File)