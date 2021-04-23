from tkinter import *
from PIL import Image, ImageTk
import threading
import time

import dlib
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from vgg16 import define_model_vgg16
import torchvision.transforms as transforms



torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

root = Tk()
pos =1
flag_happy = False
flag_fear = False
flag_surprise = False
flag_disgust = False
flag_sad = False
flag_angry = False
mode=0
teacher_mode_flag = False

happy = IntVar()
fear = IntVar()
surprise = IntVar()
disgust = IntVar()
sad = IntVar()
angry = IntVar()
happy.set("0")
fear.set("0")
surprise.set("0")
disgust.set("0")
sad.set("0")
angry.set("0")


#Database
import sqlite3

#Create a database or connect to database
conn = sqlite3.connect('AI_database.db')

#Create cursor
c = conn.cursor()

# c.execute("""CREATE TABLE table_q1(
#     Happy INTEGER,
#     Fear INTEGER,
#     Surprise INTEGER,
#     Disgust INTEGER,
#     SAd INTEGER,
#     Angry INTEGER
#     )""")
#
# c.execute("""CREATE TABLE table_q2(
#     Happy INTEGER,
#     Fear INTEGER,
#     Surprise INTEGER,
#     Disgust INTEGER,
#     SAd INTEGER,
#     Angry INTEGER
#     )""")
#
# c.execute("""CREATE TABLE table_q3(
#     Happy INTEGER,
#     Fear INTEGER,
#     Surprise INTEGER,
#     Disgust INTEGER,
#     SAd INTEGER,
#     Angry INTEGER
#     )""")




fontStyle = ("Helvetica", 14)
fontHead=("Helvetica", 18,'bold')
Width = 1024
Height = 750
bg = PhotoImage(file="../images/spc.png")
label11 = Label(root, image=bg)
label11.place(x=0, y=0,width=Width,height=Height,anchor="nw")

q1 = ImageTk.PhotoImage(file="../images/AI_Q1.PNG")
q2 = ImageTk.PhotoImage(file="../images/AI_Q2.PNG")
q3 = ImageTk.PhotoImage(file="../images/AI_Q3.PNG")
Question = Label(root,image=bg,height=100,width=Width)




mainFrame = Frame(root, width=Width, height=Height)
mainFrame.pack_propagate(False)
mainFrame.pack()
label1 = Label(mainFrame, image=bg)
label1.place(x=0, y=0,relwidth=1,relheight=1,anchor="nw")

popups=Frame(mainFrame, height=460, bg='lightgrey', bd=1 )
q_no = Label(mainFrame, text="Q1/3", padx=5, pady=5)

submit=Button(mainFrame,text="Save")


root.title('Quizy')

t = IntVar()
t=15
a = Label(mainFrame, text='Welcome!',bg='#143E94',  font=fontHead,fg='white')
a.pack()

h = Label(mainFrame, text='',bg='#143E94',
          font=fontHead,fg='white')
h.pack()

teach= Label(mainFrame, text=' ',bg='#143E94', font=fontHead,fg='white')
teach.pack()

camVideo = Frame(mainFrame, width=Width, height=460, bg='#0E347B',bd=1)
camVideo.pack()
Label(camVideo, text='Webcam', bg='#0E347B',fg='white', font=fontStyle).pack()




# Quiz information frame
quizInfo = Frame(mainFrame,bg='#03081D')
quizInfo.pack()
Label(quizInfo, text='Quiz Details:', bg='#03081D',font=fontStyle, fg='white').pack()
Label(quizInfo, text='Subject Code: CS-621', bg='#03081D', font=fontStyle,fg='white').pack()
Label(quizInfo, text='Subject Name: Artificial Intelligence', bg='#03081D',font=fontStyle,fg='white').pack()
Label(quizInfo, text='Maximum Marks: 100', bg='#03081D', font=fontStyle,fg='white').pack()
Label(quizInfo, text='Duration: 1Hour 45 Min', bg='#03081D', font=fontStyle,fg='white').pack()

def updateTime():
    global h,t,teach
    while t and mode !=2:
        time.sleep(1)
        t -= 1
        if t<=5:
            h.configure(text='Smile in ' + str(t) + ' seconds :) to enter into student mode.', fg='red')
        elif t<=10 and t>5:
            teach.configure(text='Otherwise you will be logged in into teacher\'s mode')
            h.configure(text='Smile in ' + str(t) + ' seconds :) to enter into student mode.')

#Student mode
def goLeft():
    global pos,q1,q2,q3,Question,popups,submit
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry
    global happy, fear, surprise, disgust, sad, angry,q_no

    happy.set("0")
    fear.set("0")
    surprise.set("0")
    disgust.set("0")
    sad.set("0")
    angry.set("0")
    if pos>1:
        pos=pos-1
        popups.destroy()
        popups = Frame(mainFrame, height=460, bg='lightgrey', bd=1)
        popups.grid_propagate(True)
        popups.grid(row=2, column=6, columnspan=4)
        flag_happy = False
        flag_fear = False
        flag_surprise = False
        flag_disgust = False
        flag_sad = False
        flag_angry = False

    Question.forget()

    if pos==1:
        submit.configure(text='Save')
        Question = Label(mainFrame, image=q1, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q1/3")
    elif pos==2:
        submit.configure(text='Save')
        Question = Label(mainFrame, image=q2, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q2/3")
    elif pos==3:
        submit.configure(text='Submit and Exit')
        Question = Label(mainFrame, image=q3, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q3/3")



def goRight():
    global pos,q1,q2,q3,Question,popups,submit
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry
    global happy, fear, surprise, disgust, sad, angry,q_no

    happy.set("0")
    fear.set("0")
    surprise.set("0")
    disgust.set("0")
    sad.set("0")
    angry.set("0")

    if pos<3:
        pos=pos+1
        popups.destroy()
        popups = Frame(mainFrame, height=460, bg='lightgrey', bd=1)
        popups.grid_propagate(True)
        popups.grid(row=2, column=6, columnspan=4)
        flag_happy = False
        flag_fear = False
        flag_surprise = False
        flag_disgust = False
        flag_sad = False
        flag_angry = False

    Question.forget()

    if pos == 1:
        submit.configure(text='Save')
        Question = Label(mainFrame, image=q1, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q1/3")
    elif pos == 2:
        submit.configure(text='Save')
        Question = Label(mainFrame, image=q2, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q2/3")
    elif pos == 3:
        submit.configure(text='Submit and Exit')
        Question = Label(mainFrame, image=q3, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
        q_no.configure(text="Q3/3")





def submitt():
    global root,pos
    if pos==3:
        save()
        root.destroy()
    else:
        save()

text="Happy"

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, upsample_num_times = 0)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

net = define_model_vgg16(7)
net = net.type(dtype)

net.load_state_dict(torch.load("vggE100.pth.tar",map_location='cuda:0')['state_dict'])
def predict_face_expression(net,face):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.55199619], std=[0.2486985])
    ])(face)

    x = torch.unsqueeze(x, 0)
    scores = net(x.type(dtype))
    _, predictions = scores.max(1)

    if predictions == 0:
        text = "Angry"
    elif predictions == 1:
        text = "Disgust"
    elif predictions == 2:
        text = "Fear"
    elif predictions == 3:
        text = "Happy"
    elif predictions == 4:
        text = "Sad"
    elif predictions == 5:
        text = "Surprise"
    else:
        text = "Neutral"

    return text

flag_popup = True

global_text = ""
global_face_rect = [0,0,0,0]
count_frame = 0
count_smile =0
count_exp=0
cap = cv2.VideoCapture(0)
def videoLoop():
    global root
    global cap,net
    global image1
    global t,text
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry, mode, count_smile,count_exp,popups
    global happy, fear, surprise, disgust, sad, angry, teacher_mode_flag, flag_popup, count_frame, global_face_rect,global_text

    vidLabel = Label(root, anchor=W)
    vidLabel.pack(expand=YES, fill=BOTH)

    camVideoLabel = Label(camVideo,width=460, height=460, bg='white',anchor="w")
    camVideoLabel.pack()
    while t:
        if teacher_mode_flag:
            break
        ret, frame = cap.read()
        count_frame = count_frame+1

        if not (ret):
            cap.release()
            cv2.destroyAllWindows()
            root.quit()

        if count_frame%5==0:
            count_frame = 0

            image1 = frame
            detected_faces = detect_faces(frame)
            if len(detected_faces)==0:
                global_face_rect = [0,0,0,0]
                global_text = " "
            for n, face_rect in enumerate(detected_faces):
                global_face_rect = face_rect
                face = Image.fromarray(frame).crop(face_rect)
                image1 = cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 4)
                font = cv2.FONT_HERSHEY_DUPLEX

                imsize = (48, 48)
                if imsize[0] > face.size[0]:
                    im = face.resize(imsize, Image.BICUBIC)
                else:
                    im = face.resize(imsize, Image.ANTIALIAS)

                # text = "Fear"
                text = predict_face_expression(net, im)
                global_text = text
                cv2.putText(image1, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
            if text=="Happy" and t<=5:
                count_smile+=1

            count_exp=count_exp+1

            if mode==2:
                if flag_popup:
                    flag_popup = False
                    popups.grid_propagate(True)
                    popups.grid(row=2, column=6, columnspan=4)
                if text == "Happy" and (not flag_happy) :
                    flag_happy=True
                    Label(popups, text="Is the question too easy?",font=fontStyle,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=happy, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=happy, value=0,bg='lightgrey').pack(anchor="w")
                elif text == "Fear" and (not flag_fear):
                    flag_fear=True
                    Label(popups, text="Is the question too hard?",font=fontStyle,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=fear, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=fear, value=0,bg='lightgrey').pack(anchor="w")
                elif text == "Surprise" and (not flag_surprise):
                    flag_surprise=True
                    Label(popups, text="Have you seen the question somewhere?",font=fontStyle,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=surprise, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=surprise, value=0,bg='lightgrey').pack(anchor="w")
                elif text == "Disgust" and (not flag_disgust):
                    flag_disgust=True
                    Label(popups, text="Are you stuck in between the solution of the question?",font=fontStyle,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=disgust, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=disgust, value=0,bg='lightgrey').pack(anchor="w")
                elif text == "Sad" and (not flag_sad):
                    flag_sad=True
                    Label(popups, text="Are you not able to understand the question?",font=fontStyle,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=sad, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=sad, value=0,bg='lightgrey').pack(anchor="w")
                elif text == "Angry" and (not flag_angry):
                    flag_angry=True
                    Label(popups, text="Is the question out of syllabus?",font=fontStyle, bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="Yes", variable=angry, value=1,bg='lightgrey').pack(anchor="w")
                    Radiobutton(popups, text="No", variable=angry, value=0,bg='lightgrey').pack(anchor="w")
            frame = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            camVideoLabel.configure(image=frame)
            camVideoLabel.image = frame
        else:
            image1 = cv2.rectangle(frame, (global_face_rect[0], global_face_rect[1]), (global_face_rect[2], global_face_rect[3]), (0, 255, 0), 4)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image1, global_text, (global_face_rect[0] + 6, global_face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
            frame = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            camVideoLabel.configure(image=frame)
            camVideoLabel.image = frame

    root.quit()



def save():
    global pos
    global happy, fear, surprise, disgust, sad, angry

    conn = sqlite3.connect('AI_database.db')
    c = conn.cursor()

    if pos ==1:
        c.execute("INSERT INTO table_q1 VALUES (:happy,:fear,:surprise,:disgust,:sad,:angry)",
                  {
                      'happy': happy.get(),
                      'fear': fear.get(),
                      'surprise': surprise.get(),
                      'disgust': disgust.get(),
                      'sad': sad.get(),
                      'angry': angry.get()
                  })
    elif pos ==2:
        c.execute("INSERT INTO table_q2 VALUES (:happy,:fear,:surprise,:disgust,:sad,:angry)",
                  {
                      'happy': happy.get(),
                      'fear': fear.get(),
                      'surprise': surprise.get(),
                      'disgust': disgust.get(),
                      'sad': sad.get(),
                      'angry': angry.get()
                  })
    else:
        c.execute("INSERT INTO table_q3 VALUES (:happy,:fear,:surprise,:disgust,:sad,:angry)",
                  {
                      'happy': happy.get(),
                      'fear': fear.get(),
                      'surprise': surprise.get(),
                      'disgust': disgust.get(),
                      'sad': sad.get(),
                      'angry': angry.get()
                  })

    conn.commit()
    conn.close()

def studentMode():
    global mainFrame,bg,Width,Height,root,pos,left,right,Question,q1,q2,q3,camVideo,t,popups,mode,submit, bg, q_no
    mode=2
    t=True
    mainFrame.destroy()
    mainFrame = Frame(root, width=Width, height=Height)
    mainFrame.pack_propagate(False)
    mainFrame.pack()

    if teacher_mode_flag:

        bg = PhotoImage(file="../images/spc.png")
        label11 = Label(mainFrame, image=bg)
        label11.place(x=0, y=0, width=Width, height=Height, anchor="nw")

        query_btn = Button(label11, text="Show Records", command=query,padx = 10,pady=10)
        query_btn.place(relx=0.5, rely=0.5, anchor=CENTER)

    else:
        label1 = Label(mainFrame, bg='lightgrey')
        label1.place(x=0, y=0, relwidth=1, relheight=1, anchor="nw")

        Question = Label(mainFrame,image=q1,height=200,width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1,column=0,columnspan=10)

        left = Button(mainFrame, text="<<", padx=5, pady=5, command=goLeft)
        left.grid(row=0, column=0)
        right = Button(mainFrame, text=">>", padx=5, pady=5, command=goRight)
        right.grid(row=0, column=10, columnspan=5)

        q_no = Label(mainFrame, text="Q1/3", padx=5, pady=5)
        q_no.grid(row=0, column=5, columnspan=1)

        camVideo = Frame(mainFrame, height=360, bg='#0E347B',padx=2,pady=2, bd=5,)
        camVideo.grid_propagate(True)
        videoThread = threading.Thread(target=videoLoop, args=())
        videoThread.start()
        camVideo.grid(row=2,column=0,columnspan=6)
        popups = Frame(mainFrame, height=460, bg='lightgrey', bd=1)

        popups.grid_propagate(True)
        popups.grid(row=2, column=6, columnspan=4)



        submit=Button(mainFrame,text="Save",command=submitt)
        submit.grid(row=3,column=5)


def query():
    conn = sqlite3.connect('AI_database.db')
    c = conn.cursor()

    c.execute("Select * From table_q1")
    records_1 = c.fetchall()

    c.execute("Select * From table_q2")
    records_2 = c.fetchall()
    c.execute("Select * From table_q3")
    records_3 = c.fetchall()

    conn.commit()
    conn.close()

    tab1 = records_1
    tab2 = records_2
    tab3 = records_3

    # data to plot
    n_groups = 6
    row1 = len(tab1)
    row2 = len(tab2)
    row3 = len(tab3)

    percentage1 = (0, 0, 0, 0, 0, 0)
    percentage2 = (0, 0, 0, 0, 0, 0)
    percentage3 = (0, 0, 0, 0, 0, 0)

    for ii in range(0, row1):
        percentage1 = tuple(map(lambda i, j: i + j, tab1[ii], percentage1))

    for ii in range(0, row2):
        percentage2 = tuple(map(lambda i, j: i + j, tab2[ii], percentage2))
    for ii in range(0, row3):
        percentage3 = tuple(map(lambda i, j: i + j, tab3[ii], percentage3))

    prod1 = tuple(i * 100 / row1 for i in percentage1)
    prod2 = tuple(i * 100 / row2 for i in percentage2)
    prod3 = tuple(i * 100 / row3 for i in percentage3)

    q1 = prod1
    q2 = prod2
    q3 = prod3

    # create plot
    fig, ax = plt.subplots(figsize=(10,5))

    index = np.arange(n_groups)
    bar_width = 0.3
    opacity = 0.6

    rects1 = plt.bar(index, q1, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Q1')

    rects2 = plt.bar(index + bar_width, q2, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Q2')

    rects3 = plt.bar(index + bar_width + bar_width, q3, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Q3')

    plt.xlabel(" ")
    plt.ylabel('Number of Students')
    plt.title('Feedback on Quiz questions')
    plt.xticks(index, ('Too easy', 'Too hard','Seen somewhere','Stuck in solution','Not understandable','Out of syllabus'))
    plt.legend()

    plt.tight_layout()
    plt.show()




def change_screen():
    global count_smile,count_exp,videoThread,t, teacher_mode_flag
    while True:
        if(t==0):

            if (count_smile/count_exp)>0.3:
                studentMode()
                break
            else:
                count_smile = 0

                teacher_mode_flag = True
                studentMode()
                break

videoThread = threading.Thread(target=videoLoop, args=())
videoThread.start()
timeThread = threading.Thread(target=updateTime, args=())
timeThread.start()
smileThread = threading.Thread(target=change_screen, args=())
smileThread.start()


root.mainloop()
cap.release()

#Commit changes
conn.commit()

#Close connection
conn.close()