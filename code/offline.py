import dlib
from PIL import Image
import cv2
import torch
from vgg16 import define_model_vgg16
import torchvision.transforms as transforms

dtype = torch.cuda.FloatTensor

live= False
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, upsample_num_times = 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

#Define the model
net = define_model_vgg16(7)
net = net.type(dtype)

#Load the trained model
net.load_state_dict(torch.load("vggE100.pth.tar",map_location='cuda:0')['state_dict'])


def predict_face_expression(net,face):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507395], std=[0.2551289])
    ])(face)
    x = torch.unsqueeze(x, 0)

    #Predict the facial expression
    scores = net(x.type(dtype))
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


count_frame = 0
global_face_rect = [0,0,0,0]
global_text =""

video_capture = cv2.VideoCapture("../video/uploaded_video.mp4")
result = cv2.VideoWriter('../video/output_video.mp4',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (int(video_capture.get(3)),int(video_capture.get(4))))


while True:
    ret, frame = video_capture.read()
    global image1
    image1 = frame
    count_frame +=1

    #Predict the expression on every 5th frame
    if count_frame % 10 == 0:
        count_frame = 0
        detected_faces = detect_faces(image1)
        if len(detected_faces)==0:
            global_face_rect=[0,0,0,0]
            global_text=""

        #Detect face on the frame
        for n, face_rect in enumerate(detected_faces):
            global_face_rect = face_rect

            face = Image.fromarray(frame).crop(face_rect)

            #Draw the rectangle on the frame
            image1 = cv2.rectangle(frame, (face_rect[0],face_rect[1]), (face_rect[2],face_rect[3]), (0,255,0), 4)
            font = cv2.FONT_HERSHEY_DUPLEX

            imsize = (48,48)
            if imsize[0] > face.size[0]:
                im = face.resize(imsize, Image.BICUBIC)
            else:
                im = face.resize(imsize, Image.ANTIALIAS)

            #Predict facial expression of the user
            text = predict_face_expression(net,im)
            global_text = text

            #Put text on the frame
            cv2.putText(image1, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
            cv2.imshow('Video', image1)
            result.write(image1)
        else:
            # Draw the rectangle on the frame
            image11 = cv2.rectangle(image1, (global_face_rect[0], global_face_rect[1]),
                                    (global_face_rect[2], global_face_rect[3]), (0, 255, 0),
                                    4)
            image1 = image11
            font = cv2.FONT_HERSHEY_DUPLEX
            # Put text on the frame
            cv2.putText(image1, global_text, (global_face_rect[0] + 6, global_face_rect[3] + 16), font, 0.75,
                        (0, 0, 255), 2)
            cv2.imshow('Video', image1)
            result.write(image1)


    if ret == False:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
print("The video was successfully saved")
