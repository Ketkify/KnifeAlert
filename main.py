import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import os
from ultralytics import YOLO
import imutils.video as vid
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
import threading
from playsound import playsound##

def alarm():##
    playsound("C:\\Users\\DELL\\OneDrive\\Desktop\\audio\\audio.mp3")##
# Function to handle button click
def open_video():
    # Open a file dialog to select the video file
    video_file_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mkv")])
    '''
    # Check if a file is selected
    if video_file_path:
        # Read the video file using OpenCV
        video_capture = cv2.VideoCapture(video_file_path)
        
        while True:
            # Read a frame from the video
            ret, frame = video_capture.read()
            
            # Check if frame is successfully read
            if not ret:
                break
            cv2.imshow("Video Frame", frame)
            
            # Check for key press to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        video_capture.release()
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''
    first_function(video_file_path)



# Function to be called when Button 1 is clicked
def first_function(video_file_path):
    condition=False

    VIDEOS_DIR = os.path.join('.', 'videos')

    video_path = os.path.join(VIDEOS_DIR,video_file_path)
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

    # Load a model
    model = YOLO("C:\\Users\\DELL\\OneDrive\\Desktop\\objectDetect\\yolov8n.pt")  # load a custom model

    threshold = 0.5

    while ret:

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                if results.names[int(class_id)] == 'knife':#
                    condition =True
                    second_function(video_file_path)
                    
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    # Play the output video
    player = vid.FileVideoStream(video_path_out).start()
    while True:
        frame = player.read()
        if frame is None:
            break
        cv2.imshow("Output Video", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    player.stop()
  
        

# Second function
def second_function(video_file_path):
    df = pd.read_csv("C:\\Users\\DELL\\OneDrive\\Desktop\\objectDetect\\superfinaldata.csv")
    x = df.drop('class', axis=1) #Features #stores all the columns in x except the 'class' column
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

    scalar = StandardScaler()
    scaled = scalar.fit(x_train, y_train)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # yhat = model.predict(x_test)
    # print(accuracy_score(y_test, yhat))
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_file_path)
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten()
                row = row.reshape(1, -1)
                # print(row)
                # x = pd.DataFrame([row]) # that one array of the pose and face coordinates
                attack_class = model.predict(row)[0] # we are predicting the class of that behaviour
                # print(attack_class)
                attack_prob = model.predict_proba(x)[0] # predicting the probability of being that actual class
                # print(attack_class, attack_prob)

                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, 
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y))
                            , [640,480]).astype(int))
                
                # Rendering the image 
                cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(attack_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                cv2.putText(image, attack_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
                # Display Class
                cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, attack_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display Probability
                cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(attack_prob[np.argmax(attack_prob)],2))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if attack_class.lower() == "attacking": ## 
                    threading.Thread(target=alarm).start() ##
            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def changeOnHover(button, colorOnHover, colorOnLeave):
    button.bind("<Enter>", func=lambda e: button.config(
        background=colorOnHover))
    # background color on leaving widget
    button.bind("<Leave>", func=lambda e: button.config(
        background=colorOnLeave))
# Create the main window
root = tk.Tk()
root.title("Knife Attack Prediction")
root.geometry("700x600")
# Load the background image
background_image = Image.open("C:\\Users\\DELL\\OneDrive\\Desktop\\objectDetect\\robo.jpg")  
background_photo = ImageTk.PhotoImage(background_image)

# Create a label for the background image and place it at the back
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)
pillow_image = Image.open("C:\\Users\\DELL\\OneDrive\\Desktop\\objectDetect\\knife_attack.jpg")  

# Convert the Pillow image to a Tkinter compatible image format
tk_image = ImageTk.PhotoImage(pillow_image)

# Create a label to display the image on the Tkinter window
image_label = tk.Label(root, image=tk_image,height=250,width=675,borderwidth=2)
image_label.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

button1 = tk.Button(root, text="Upload Video To Start Prediction", command=open_video, height=3, width=30, borderwidth=2)
button1.place(relx=0.5, rely=0.75, anchor=tk.CENTER)

changeOnHover(button1, "red", "white")

root.mainloop()
