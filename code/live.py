import dlib
from PIL import Image
import cv2
import torch
from vgg16 import define_model_vgg16
import torchvision.transforms as transforms

dtype = torch.cuda.FloatTensor

# live= False
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, upsample_num_times = 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

net = define_model_vgg16(7)
net = net.type(dtype)
net.load_state_dict(torch.load("vggE100.pth.tar",map_location='cuda:0')['state_dict'])


def predict_face_expression(net,face):
    x = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507395], std=[0.2551289])
    ])(face)
    x = torch.unsqueeze(x, 0)
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

print('Turning on webcam')
video = cv2.VideoCapture(0)
if (video.isOpened() == False):
    print("Error reading video file")
result = cv2.VideoWriter('../video/saved_live.mp4',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (int(video.get(3)),int(video.get(4))))
while (True):
    #Take one frame from the video
    ret, frame = video.read()
    image11= frame

    if ret == True:

        #Predict expression on every 5th frame
        if count_frame%5==0:
            count_frame=0

            #Detect the face in the frame
            detected_faces = detect_faces(frame)
            for n, face_rect in enumerate(detected_faces):
                global_face_rect = face_rect
                face = Image.fromarray(frame).crop(face_rect)

                #Draw rectangle on the face
                image11 = cv2.rectangle(frame, (face_rect[0], face_rect[1]), (face_rect[2], face_rect[3]), (0, 255, 0),4)
                font = cv2.FONT_HERSHEY_DUPLEX

                imsize = (48, 48)
                if imsize[0] > face.size[0]:
                    im = face.resize(imsize, Image.BICUBIC)
                else:
                    im = face.resize(imsize, Image.ANTIALIAS)

                #Predict the expression of the user
                text = predict_face_expression(net, im)
                global_text = text

                #Draw text on the frame
                cv2.putText(image11, text, (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)

        else:
            count_frame = count_frame+1

            # Draw rectangle on the face
            image11 = cv2.rectangle(frame, (global_face_rect[0], global_face_rect[1]), (global_face_rect[2], global_face_rect[3]), (0, 255, 0),
                                    4)
            font = cv2.FONT_HERSHEY_DUPLEX
            # Draw text on the frame
            cv2.putText(image11, global_text, (global_face_rect[0] + 6, global_face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
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
