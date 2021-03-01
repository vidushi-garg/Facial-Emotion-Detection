import dlib
from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
import cv2
live= True
def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

if not live:
    video_capture = cv2.VideoCapture("data/video3.mp4")
    result = cv2.VideoWriter('data/output3.mp4',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             10, (int(video_capture.get(3)),int(video_capture.get(4))))
    while True:
        ret, frame = video_capture.read()
        image1 = frame
        detected_faces = detect_faces(frame)
        for n, face_rect in enumerate(detected_faces):
            face = Image.fromarray(frame).crop(face_rect)
            image1 = cv2.rectangle(frame, (face_rect[0],face_rect[1]), (face_rect[2],face_rect[3]), (0,255,0), 4)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image1, 'HAPPY', (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
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
    video = cv2.VideoCapture(1)
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
                cv2.putText(image11, 'HAPPY', (face_rect[0] + 6, face_rect[3] + 16), font, 0.75, (0, 0, 255), 2)
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