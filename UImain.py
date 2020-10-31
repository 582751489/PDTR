from tkinter import *
from tkinter import filedialog
from tkinter import ttk
import cv2
import numpy as np
import os


class ButtonInput(Button):
    def __init__(self, frame):
        self.btn = Button(frame, text='+', command=self.get_dict, font=("Times", "12"))
        self.btn.pack(side=LEFT)
        self.dict = ''

    def get_dict(self):
        self.dict = str(filedialog.askopenfilenames(initialdir=os.path.dirname(__file__))[0])


def run_input(file1, file2, file3):
    """
    依次执行命令行 -> 保存运行结果 -> 播放视频
    :param file1: camera 1 路径
    :param file2: camera 2 路径
    :param file3: camera 3 路径
    :return:
    """
    # 执行main
    os.system('python main.py -i ' + file1 + ' -camera c1')
    os.system('python main.py -i ' + file2 + ' -camera c2')
    os.system('python main.py -i ' + file3 + ' -camera c3')
    # # 执行test
    os.system('python test.py --name PCB --test_dir .\person')

    # 播放本地保存视频
    names = [r'.\output\output1.avi',
             r'.\output\output2.avi',
             r'.\output\output3.avi',
             ]
    window_titles = ['Camera 1',
                     'Camera 2',
                     'Camera 3',
                     ]
    reshape = (500, 500)  # 视频输出尺寸
    for i in range(len(names)):
        cap = cv2.VideoCapture(names[i])
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.resize(frame, reshape)
            cv2.waitKey(20)
            cv2.imshow(window_titles[i], frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


res1_id = ''
res2_id = ''
res3_id = ''


def run_reid(camera, person_id):
    """
    运行Re-ID程序
    :param camera: 摄像头ID
    :param person_id: 行人ID
    :return:
    """
    global res1_id, res2_id, res3_id
    if camera == '1':
        res1_id = person_id
    elif camera == '2':
        res2_id = person_id
    elif camera == '3':
        res3_id = person_id

    r = os.popen('python demo.py --query_index ' + str(person_id) + ' --camera c' + str(camera))
    output = r.readlines()
    for line in output:
        id_now = re.search('ID:\d{1,9}', line)
        camera_now = re.search('Camera:c\d{1,9}', line)
        if id_now and camera_now:
            id_now = id_now.group(0).strip('ID:')
            camera_now = camera_now.group(0).strip('Camera:c')
            if camera_now == '1':
                res1_id = id_now
            elif camera_now == '2':
                res2_id = id_now
            elif camera_now == '3':
                res3_id = id_now
        else:
            pass


def res(c1_index, c2_index, c3_index):
    """
    输出TimeTable & PersonTable
    :return:
    """
    # 运行并保存
    os.system('python vis.py --C1_index ' + c1_index + ' --C2_index ' + c2_index + ' --C3_index ' + c3_index)

    # 输出TimeTable & PersonTable
    timetable = cv2.imread(r'.\timetable.png')
    persontable = cv2.imread(r'.\persontable.png')
    cv2.imshow('TimeTable', timetable)
    cv2.imshow('PersonTable', persontable)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def res_tracking(dict_c1, dict_c2, dict_c3, c1_index, c2_index, c3_index):
    """
    依次输出跟踪视频
    :return:
    """
    # 保存跟踪视频
    os.system('python main.py -i ' + dict_c1 + ' -camera c1 -ids ' + c1_index)
    os.system('python main.py -i ' + dict_c2 + ' -camera c2 -ids ' + c2_index)
    os.system('python main.py -i ' + dict_c3 + ' -camera c3 -ids ' + c3_index)

    # 输出结果
    names = [r'.\output\output1_reid.avi',
             r'.\output\output2_reid.avi',
             r'.\output\output3_reid.avi']
    window_titles = ['Camera 1 Re-ID', 'Camera 2 Re-ID', 'Camera 3 Re-ID',
                     ]
    reshape = (500, 500)  # 视频输出尺寸
    for i in range(len(names)):
        cap = cv2.VideoCapture(names[i])
        while cap.isOpened():
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = cv2.resize(frame, reshape)
            cv2.waitKey(20)
            cv2.imshow(window_titles[i], frame)
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


# 新建窗口和布局
window = Tk()
window.title("Eyes of God System")
window.geometry("700x600")

# 输入视频
# 容器
frame_input = Frame(window)
frame_input.pack(side=TOP, anchor=NW, ipady=20)

# Camera 1
# 创建容器
frame_c1 = Frame(frame_input)
frame_c1.pack(side=TOP, anchor=NW, ipady=5)
label_reminder1 = Label(frame_c1, text="Please Load 3 Video Flies: ", font=("Times", "16"), anchor=NW)
label_reminder1.pack(side=TOP, anchor=NW)
btn_c1 = ButtonInput(frame_c1)
label_c1 = Label(frame_c1, text="Camera1 File: ", font=("Times", "16"), anchor=NW)
label_c1.pack(side=LEFT)

# 显示文件路径
label_file1 = Label(frame_c1, text=btn_c1.dict, font=("Times", "16"))
label_file1.pack(side=LEFT)

# Camera 2
# 创建容器
frame_c2 = Frame(frame_input)
# frame_c2.grid(column=0, row=1, sticky=W)
frame_c2.pack(side=TOP, anchor=NW, ipady=5)
btn_c2 = ButtonInput(frame_c2)
label_c2 = Label(frame_c2, text="Camera2 File: ", font=("Times", "16"), anchor=NW)
label_c2.pack(side=LEFT)
# 显示文件路径
label_file2 = Label(frame_c2, text=btn_c2.dict, font=("Times", "16"))
label_file2.pack(side=LEFT)

# Camera 3
# 创建容器
frame_c3 = Frame(frame_input)
# frame_c3.grid(column=0, row=2, sticky=W)
frame_c3.pack(side=TOP, anchor=NW, ipady=5)
btn_c3 = ButtonInput(frame_c3)
label_c3 = Label(frame_c3, text="Camera3 File: ", font=("Times", "16"), anchor=NW)
label_c3.pack(side=LEFT)
# 显示文件路径
label_file3 = Label(frame_c3, text=btn_c3.dict, font=("Times", "16"))
label_file3.pack(side=LEFT)

# 运行按钮
btn_input = Button(frame_input, text='Run Detection & Tracking',
                   command=lambda: run_input(btn_c1.dict, btn_c2.dict, btn_c3.dict),
                   font=("Times", "16"))
btn_input.pack(side=LEFT)
# btn_input = ButtonRun(frame_input, file1=btn_c1.dict, file2=btn_c2.dict, file3=btn_c3.dict)

# Re-id 模块
# 次级容器
frame_reid = Frame(window)
frame_reid.pack(side=TOP, anchor=NW, ipady=5)

# camera选项
frame_cam = Frame(frame_reid)
frame_cam.pack(side=TOP, anchor=NW, ipady=5)
label_cam = Label(frame_cam, text="Camera: ", font=("Times", "16"), anchor=NW)
label_cam.pack(side=LEFT)
# 下拉框
# 使用find_camera.get()获取目前的输出
find_camera = ttk.Combobox(frame_cam, values=["1", "2", "3"])
find_camera.pack(side=LEFT)

# id选项
frame_id = Frame(frame_reid)
frame_id.pack(side=TOP, anchor=NW, ipady=5)
label_id = Label(frame_id, text="Person ID: ", font=("Times", "16"), anchor=NW)
label_id.pack(side=LEFT)
# 输入框 使用find_id.get()
find_id = Entry(frame_id)
find_id.pack(side=LEFT)

# 运行re-id按钮
btn_reid = Button(frame_reid, text='Run Re-id',
                  command=lambda: run_reid(camera=find_camera.get(), person_id=find_id.get()), font=("Times", "16"))
btn_reid.pack(anchor=NW)

# 显示re-id结果

# 提示显示结果
frame_reminder = Frame(frame_reid)
frame_reminder.pack(side=TOP, anchor=NW)
reminder = Label(frame_reminder, text="Re-ID result: ", font=("Times", "16"), anchor=NW)
reminder.pack(side=LEFT)
# 结果1
frame_res1 = Frame(frame_reid)
frame_res1.pack(side=TOP, anchor=NW)
label_reminder2 = Label(frame_res1, text="Camera 1 : ", font=("Times", "16"), anchor=NW)
label_reminder2.pack(side=LEFT)
label_res1 = Label(frame_res1, text=" ", font=("Times", "16"), anchor=NW)
label_res1.pack(side=LEFT, anchor=NW)
# 结果2
frame_res2 = Frame(frame_reid)
frame_res2.pack(side=TOP, anchor=NW)
label_reminder3 = Label(frame_res2, text="Camera 2 : ", font=("Times", "16"), anchor=NW)
label_reminder3.pack(side=LEFT)
label_res2 = Label(frame_res2, text=" ", font=("Times", "16"), anchor=NW)
label_res2.pack(side=LEFT, anchor=NW)
# 结果3
frame_res3 = Frame(frame_reid)
frame_res3.pack(side=TOP, anchor=NW)
label_reminder4 = Label(frame_res3, text="Camera 3 : ", font=("Times", "16"), anchor=NW)
label_reminder4.pack(side=LEFT)
label_res3 = Label(frame_res3, text=" ", font=("Times", "16"), anchor=NW)
label_res3.pack(side=TOP, anchor=NW)

# 输出部分
frame_output = Frame(window)
frame_output.pack(side=TOP, ipady=10)
btn_timetable = Button(frame_output, text='TimeTable & PersonTable', command=lambda: res(res1_id, res2_id, res3_id),
                       font=("Times", "16"), )
btn_timetable.pack(side=LEFT, padx=20)
btn_tracking = Button(frame_output, text='Tracking Video',
                      command=lambda: res_tracking(btn_c1.dict, btn_c2.dict, btn_c3.dict, res1_id, res2_id, res3_id),
                      font=("Times", "16"))
btn_tracking.pack(side=LEFT, padx=20)


def refresh_data():
    # 更新文件路径显示
    label_file1.config(text=btn_c1.dict)
    label_file2.config(text=btn_c2.dict)
    label_file3.config(text=btn_c3.dict)
    label_res1.config(text=res1_id)
    label_res2.config(text=res2_id)
    label_res3.config(text=res3_id)
    window.after(100, func=refresh_data)


window.after(100, func=refresh_data)
window.mainloop()
