'''
运行后展示登录页面，可先注册用户名，登录后，点击'选择图像'后，跳到打开文件open_file.py
'''
import pickle  # 存放数据
from open_file import *

# 1.创建窗口（window）
window = tk.Tk()
# 2.设置这个窗口（window）的相关信息
window.title("甲状腺结节超声图像分割系统---欢迎页面")
window.geometry("498x350") # 长498，高350

def login():
    # print('login')
    usr_name = var_usr_name.get()  # 获取用户名输入框中，用户名。 usr_name此时获得的就是一个普通的string类型
    usr_pwd = var_usr_pwd.get()  # 获取密码输入框中，密码
    try:
        with open('usrs_info.pickle', 'rb') as usr_file:
            usrs_info = pickle.load(usr_file)
    except FileNotFoundError:
        with open('usrs_info.pickle', 'wb') as usr_file:
            usrs_info = {'admin': 'admin'}
            pickle.dump(usrs_info, usr_file)

    is_sign_up = False
    if usr_name in usrs_info:
        if usr_pwd == usrs_info[usr_name]:
            # 在此处调用open_file.py函数
            # messagebox.showinfo(title='Welcome', message='How are you? ' + usr_name)

            window_upload = tk.Toplevel(window)  # toplevel（顶级窗口），类似于弹出窗口，具有独立的窗口属性（如标题栏、边框等）
            window_upload.geometry('250x250')
            window_upload.title("选择图像页面")

            # label展示文字
            label_text = tk.Label(window_upload, justify='left', padx=10, text="""您已登录成功，请点击下方按钮选择图像:""").place(x=10, y=50)

            upload = tk.Button(window_upload, text='选择图像', command=lambda :open_file(window_upload)) # 设置字体和大小, font=('Arial', 12), width=10, height=1
            upload.place(x=100, y=100)

        else:
            messagebox.showerror(message="Error! your password is wrong!")
    else:
        is_sign_up = messagebox.askyesno(title="Welcome", message='You have not sign up yet. Do you want to sign up?')

    if is_sign_up:
        sign_up()


def sign_up():
    def sign_to():
        # 保存所有输入的信息
        np = new_pwd.get()
        npf = new_pwd_conf.get()
        nn = new_name.get()
        try:
            with open('usrs_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)

        except FileNotFoundError:
            with open('usrs_info.pickle', 'wb') as usr_file:
                usrs_info = {'admin': 'admin'}
                pickle.dump(usrs_info, usr_file)
            with open('usrs_info.pickle', 'rb') as usr_file:
                exist_usr_info = pickle.load(usr_file)

        if np == '':
            messagebox.showerror("Error!", "Password is none!")
        elif np != npf:
            messagebox.showerror("Error!", "Password and confirm password must be the same!")
        elif nn in exist_usr_info:
            messagebox.showerror("Error!", "The user has already exit!")
        else:
            exist_usr_info[nn] = np
            with open('usrs_info.pickle', 'wb') as usr_file:
                pickle.dump(exist_usr_info, usr_file)  # 存入usrs_info.pickle文件
                messagebox.showinfo('Welcome', 'You have signed up!')
                # 关闭这个顶级窗口
                window_sign_up.destroy()

    # print('sign up')
    # 另外创建一个窗口，采用toplevel的形式
    window_sign_up = tk.Toplevel(window)  # toplevel（顶级窗口），类似于弹出窗口，具有独立的窗口属性（如标题栏、边框等）
    # 设置这个弹出窗口（windo_sign_up）的相关属性
    window_sign_up.geometry('350x200')
    window_sign_up.title('Sign up window')

    # 在顶级窗口中设置相应的注册组件
    # 用户名 username
    new_name = tk.StringVar()
    new_name.set('doctor')
    tk.Label(window_sign_up, text='Username').place(x=70, y=50)
    tk.Entry(window_sign_up, textvariable=new_name).place(x=150, y=50)

    # 密码 password
    new_pwd = tk.StringVar()
    tk.Label(window_sign_up, text="password").place(x=70, y=89)
    tk.Entry(window_sign_up, textvariable=new_pwd, show='*').place(x=150, y=89)
    # 确认密码
    new_pwd_conf = tk.StringVar()
    tk.Label(window_sign_up, text="confirm").place(x=70, y=119)
    tk.Entry(window_sign_up, textvariable=new_pwd_conf, show="*").place(x=150, y=119)

    # 确认button
    tk.Button(window_sign_up, text='ok', width=10, command=sign_to).place(x=120, y=150)


# 3.各类组件
# 3.1 canvas 放置图片
# welcome image
canvas = tk.Canvas(window, height=235, width=500)
image_file = tk.PhotoImage(file='welcome-hi.gif') # 此图大小498 x 286
image = canvas.create_image(0, 0, anchor='nw', image=image_file)
canvas.pack(side='top')

# 3.2 label
# user information entry
tk.Label(window, text='Username:').place(x=50, y=250)
tk.Label(window, text='Password:').place(x=50, y=290)

# 输入用户名
var_usr_name = tk.StringVar()  # tk中的string类型
# 让用户名设置成默认值
var_usr_name.set("doctor")
entry_user_name = tk.Entry(window, textvariable=var_usr_name)  # 用户输入的信息存在var_usr_name里
entry_user_name.place(x=120, y=252)

# 输入密码
var_usr_pwd = tk.StringVar()  # password也设置成tk中的string类型
entry_user_pwd = tk.Entry(window, textvariable=var_usr_pwd, show='*')  # 用户输入的信息存在var_usr_pwd里
entry_user_pwd.place(x=120, y=292)

# 3.4 button
# login and sign up
btn_login = tk.Button(window, text='login', command=login)
btn_sign_up = tk.Button(window, text='sign up', command=sign_up)
btn_login.place(x=290, y=285)
btn_sign_up.place(x=339, y=284)

# 4.运行窗口
window.mainloop()