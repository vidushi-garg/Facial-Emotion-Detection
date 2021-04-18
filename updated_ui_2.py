from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
# from faceDetect import *
import time

import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import cv2
import torch
from vgg16 import define_model_vgg16
import torchvision.transforms as transforms

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype1 = torch.cuda.LongTensor

# cap = cv2.VideoCapture(0)

pos =1
flag_happy = False
flag_fear = False
flag_surprise = False
flag_disgust = False
flag_sad = False
flag_angry = False
mode=0





root = Tk()
fontStyle = ("Helvetica", 14)
fontHead=("Helvetica", 18,'bold')
Width = 1024
Height = 750
# style = ttk.Style(root)
# style.theme_use('classic')
bg = PhotoImage(file="../images/spc.png")
label11 = Label(root, image=bg)
label11.place(x=0, y=0,width=Width,height=Height,anchor="nw")
# Show image using label

q1 = ImageTk.PhotoImage(file="../images/AI_Q1.PNG")
q2 = ImageTk.PhotoImage(file="../images/AI_Q2.PNG")
q3 = ImageTk.PhotoImage(file="../images/AI_Q3.PNG")
Question = Label(root,image=bg,height=100,width=Width)

mainFrame = Frame(root, width=Width, height=Height)
mainFrame.pack_propagate(False)
mainFrame.pack()
label1 = Label(mainFrame, image=bg)
label1.place(x=0, y=0,relwidth=1,relheight=1,anchor="nw")
# label1.pack()

# my_canvas = Canvas(mainFrame,width = Width,height=Height)
# my_canvas.pack(fill="both",expand=True)
# my_canvas.create_image(0,0,image=bg,anchor="nw")

root.title('Quizy')

# info = Frame(mainFrame, height=120,bg="#ffffff")
# info.pack_propagate(False)
# info.pack()
t = IntVar()
t=20
Label(mainFrame, text='Welcome!',bg='#143E94',  font=fontHead,fg='white').pack()

h = Label(mainFrame, text='Smile in ' + str(t) + ' seconds :) to enter into student mode.',bg='#143E94',
          font=fontHead,fg='white')
h.pack()

Label(mainFrame, text='Otherwise you will be logged in into teacher\'s mode',bg='#143E94', font=fontHead,fg='white').pack()

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
    global h,t
    while t:
        time.sleep(1)
        t -= 1
        if t<=5:
            h.configure(text='Smile in ' + str(t) + ' seconds :) to enter into student mode.', fg='red')
        else:
            h.configure(text='Smile in ' + str(t) + ' seconds :) to enter into student mode.')

#Student mode
def goLeft():
    global pos,q1,q2,q3,Question,popups
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry

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
        Question = Label(mainFrame, image=q1, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
    elif pos==2:
        Question = Label(mainFrame, image=q2, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
    elif pos==3:
        Question = Label(mainFrame, image=q3, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)



def goRight():
    global pos,q1,q2,q3,Question,popups
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry


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
        Question = Label(mainFrame, image=q1, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
    elif pos == 2:
        Question = Label(mainFrame, image=q2, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)
    elif pos == 3:
        Question = Label(mainFrame, image=q3, height=200, width=Width)
        Question.grid_propagate(False)
        Question.grid(row=1, column=0, columnspan=10)





def submitt():
    # root.wm_protocol("WM_DELETE_WINDOW", packup)
    #
    global root
    root.destroy()
    # stop.set()

# def packup():
#     # stop.set()
#     # self.videoStream.release()
#     cv2.destroyAllWindows()
#     root.quit()


text="Happy"

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, upsample_num_times = 0)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

def predict_face_expression(net,face):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507395], std=[0.2551289])
    ])(face)
    net.load_state_dict(torch.load("vggE100.pth.tar",map_location='cuda:0')['state_dict'])
    x = torch.unsqueeze(x, 0)
    scores = net(x.type(dtype))
    # scores = net(x)
    _, predictions = scores.max(1)
    text = ""

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


cap = cv2.VideoCapture(0)
net = define_model_vgg16(7)
net = net.type(dtype)
def videoLoop():
    global root
    global cap,net
    global image1
    global t,text
    global flag_happy, flag_fear, flag_surprise, flag_disgust, flag_sad, flag_angry, mode

    vidLabel = Label(root, anchor=NW)
    vidLabel.pack(expand=YES, fill=BOTH)

    camVideoLabel = Label(camVideo,width=460, height=460, bg='white')
    camVideoLabel.pack()
    while t:
        ret, frame = cap.read()
        # print("Frame " +frame)

        if not (ret):
            cap.release()
            cv2.destroyAllWindows()
            root.quit()

        image1 = frame
        detected_faces = detect_faces(frame)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(frame).crop(face_rect)
            image1 = cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0), 4)
            font = cv2.FONT_HERSHEY_DUPLEX

            imsize = (48, 48)
            if imsize[0] > face.size[0]:
                im = face.resize(imsize, Image.BICUBIC)
            else:
                im = face.resize(imsize, Image.ANTIALIAS)

            # text = "Happy"
            text = predict_face_expression(net, im)
            cv2.putText(image1, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
        if mode==2:
            if text == "Happy" and (not flag_happy) :
                flag_happy=True
                happy = IntVar()
                happy.set("0")
                Label(popups, text="Is the question too easy?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=happy, value=1).pack()
                Radiobutton(popups, text="No", variable=happy, value=0).pack()
            elif text == "Fear" and (not flag_fear):
                flag_fear=True
                fear = IntVar()
                fear.set("0")
                Label(popups, text="Is the question too hard?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=fear, value=1).pack()
                Radiobutton(popups, text="No", variable=fear, value=0).pack()
            elif text == "Surprise" and (not flag_surprise):
                flag_surprise=True
                surprise = IntVar()
                surprise.set("0")
                Label(popups, text="Have you seen the question somewhere?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=surprise, value=1).pack()
                Radiobutton(popups, text="No", variable=surprise, value=0).pack()
            elif text == "Disgust" and (not flag_disgust):
                flag_disgust=True
                disgust = IntVar()
                disgust.set("0")
                Label(popups, text="Are you stuck in between the solution of the question?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=disgust, value=1).pack()
                Radiobutton(popups, text="No", variable=disgust, value=0).pack()
            elif text == "Sad" and (not flag_sad):
                flag_sad=True
                sad = IntVar()
                sad.set("0")
                Label(popups, text="Are you not able to understand the question?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=sad, value=1).pack()
                Radiobutton(popups, text="No", variable=sad, value=0).pack()
            elif text == "Angry" and (not flag_angry):
                flag_angry=True
                angry = IntVar()
                angry.set("0")
                Label(popups, text="Is the question out of syllabus?",font=fontStyle).pack()
                Radiobutton(popups, text="Yes", variable=angry, value=1).pack()
                Radiobutton(popups, text="No", variable=angry, value=0).pack()
        frame = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = ImageTk.PhotoImage(frame)
        camVideoLabel.configure(image=frame)
        camVideoLabel.image = frame

    root.quit()




def studentMode():
    global mainFrame,bg,Width,Height,root,pos,left,right,Question,q1,q2,q3,camVideo,t,popups,mode
    mode=2
    t=True
    mainFrame.destroy()
    mainFrame = Frame(root, width=Width, height=Height)
    mainFrame.pack_propagate(False)
    mainFrame.pack()


    label1 = Label(mainFrame, image=bg)
    label1.place(x=0, y=0, relwidth=1, relheight=1, anchor="nw")



    # if (pos==1):
    Question = Label(mainFrame,image=q1,height=200,width=Width)
    Question.grid_propagate(False)
    Question.grid(row=1,column=0,columnspan=10)

    left = Button(mainFrame, text="<<", padx=5, pady=5, command=goLeft)
    left.grid(row=0, column=0)
    right = Button(mainFrame, text=">>", padx=5, pady=5, command=goRight)
    right.grid(row=0, column=10, columnspan=5)

    camVideo = Frame(mainFrame, height=360, bg='#0E347B',padx=2,pady=2, bd=5,)
    camVideo.grid_propagate(True)
    videoThread = threading.Thread(target=videoLoop, args=())
    videoThread.start()
    camVideo.grid(row=2,column=0,columnspan=6)
    popups = Frame(mainFrame, height=460, bg='lightgrey', bd=1 )
    popups.grid_propagate(True)
    popups.grid(row=2, column=6, columnspan=4)



    submit=Button(mainFrame,text="Submit",command=submitt)
    submit.grid(row=3,column=5)




studentMode()

root.mainloop()
cap.release()