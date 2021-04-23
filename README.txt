Dependencies:
Python3
torchvision - 0.9.0.dev20210118 version
Pillow - 8.1.0 version
dlib - 19.21.1 version
Pytorch with cuda version
opencv-python - 4.5.1.48 version

To train the model, run:
python3 code/main.py
The model will be saved in the 'code' directory with the name "checkpoint.pth.tar" after every epoch

Train the model and save it in the 'code' directory with the name 'vggE100.pth.tar'

To give video as input and get annotated video with predicted facial expression as result, run:
python3 code/offine.py
Press 'q' to exit the window

Create a 'video' directory in the main directory and put the input video in the 'video' directory with the name 'uploaded_video.mp4'
The output video will be saved in 'video' directory with name 'output_video.mp4'

To get the prediction done on the live video, run:
python3 code/live.py
Press 'q' to exit the window
The output video will be saved in 'video' directory with name 'saved_live.mp4'

To run the Graphical User Interface, run:
pyhton3 code/ui.py

Dataset has been taken from : https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Put the datatset csv in 'data' folder with the name 'emotion_dataset.csv'. The 'data' folder contains the cleaned dataset csv.