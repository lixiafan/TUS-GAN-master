'''
定义open_file函数，用于打开待分割的照片
打开照片后，设置阈值，
定义按钮'上传分割'，用于调用test_modle.py，传入待分割的照片和阈值
'''
from tkinter import *
from tkinter import filedialog,messagebox
import tkinter as tk
from test_modle import test_modle

def open_file(root):
    choose_path = "./test_img"
    File = filedialog.askopenfilename(parent=root, initialdir=choose_path,title='选择一张照片')
    # 第1步，实例化object，建立窗口window
    window = tk.Toplevel()

    # 第2步，给窗口的可视化起名字
    window.title('确定选择图像并上传')

    # 第3步，设定窗口的大小(长 * 宽)
    window.geometry('400x400')  # 这里的乘是小x

    # label展示文字
    label_text = tk.Label(window, justify='left', padx=10, text="""请确认图像并设置阈值后点击'上传分割'""").place(x=70, y=20)
    # 读取图片
    logo = tk.PhotoImage(file=f'{File}')

    # label展示图片
    label_image = tk.Label(window, image=logo).place(x=70, y=50)

    # 设置阈值
    threshod = tk.StringVar()
    tk.Label(window, text="设置阈值").place(x=100, y=309)
    tk.Entry(window, textvariable=threshod).place(x=170, y=309)

    # 定义按钮，调用test_modle函数，用于分割图像
    Button(window,text='上传分割',command = lambda : test_modle(File,float(threshod.get()))).place(x=170,y=340)
    window.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    open_file(root=root)