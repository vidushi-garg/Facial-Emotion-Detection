import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import cv2
import torch
from vgg16 import define_model_vgg16
import torchvision.transforms as transforms
import PIL

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
# dtype1 = torch.cuda.LongTensor

live= True
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


if not live:
    video_capture = cv2.VideoCapture("./video/uploaded_video.mp4")
    result = cv2.VideoWriter('./video/output_video.mp4',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_capture.get(3)),int(video_capture.get(4))))


    while True:
        ret, frame = video_capture.read()
        image1 = frame
        # print(frame)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = Image.fromarray(frame)
        # frame = ImageTk.PhotoImage(frame)

        detected_faces = detect_faces(frame)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(frame).crop(face_rect)
            image1 = cv2.rectangle(frame, (face_rect[0],face_rect[1]), (face_rect[2],face_rect[3]), (0,255,0), 4)
            font = cv2.FONT_HERSHEY_DUPLEX

            # face1 = imutils.resize(face,48)
            imsize = (48,48)
            # face1 = face.resize(48,Image.BICUBIC)
            if imsize[0] > face.size[0]:
                im = face.resize(imsize, Image.BICUBIC)
            else:
                im = face.resize(imsize, Image.ANTIALIAS)

            text = predict_face_expression(net,im)
            # text = "Happy"
            cv2.putText(image1, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
        cv2.imshow('Video', image1)
        result.write(image1)
        if ret == False:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
    print("The video was successfully saved")
else:
    print('Turn on webcam')
    video = cv2.VideoCapture(0)
    if (video.isOpened() == False):
        print("Error reading video file")
    result = cv2.VideoWriter('data/live.mp4',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video.get(3)),int(video.get(4))))
    while (True):
        ret, frame = video.read()
        print(ret)
        image11= frame

        if ret == True:

            detected_faces = detect_faces(frame)
            for n, face_rect in enumerate(detected_faces):
                face = Image.fromarray(frame).crop(face_rect)
                image11 = cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0),4)
                font = cv2.FONT_HERSHEY_DUPLEX

                imsize = (48, 48)
                # face1 = face.resize(48,Image.BICUBIC)
                if imsize[0] > face.size[0]:
                    im = face.resize(imsize, Image.BICUBIC)
                else:
                    im = face.resize(imsize, Image.ANTIALIAS)

                text = predict_face_expression(net, im)

                cv2.putText(image11, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
            cv2.imshow('Video', image11)
            result.write(image11)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    video.release()
    result.release()

    cv2.destroyAllWindows()

    print("The video was successfully saved")