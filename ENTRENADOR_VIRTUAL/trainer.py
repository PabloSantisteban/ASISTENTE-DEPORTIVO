#	TRABAJO RERALIZADO POR: Rodrigo Gutiérrez Ribal (IABD05), Álex González Puente (IABD06) y Pablo Santisteban Fernández (IABD04)


# imports
import cv2                  # img and video procesment
import mediapipe as mp      # posings and landmarks
import numpy as np
import time                 
from exercises_functions import *
import sys
import platform


# set height and width of video screen
desired_width = 1920
desired_height = 1080

def calculate_angle(a, b, c):
    
    '''
    : param a: first point
    : param a: second point
    : param c: third point

    :return: angle between points a and c
    '''

    a = np.array(a)  
    b = np.array(b)  
    c = np.array(c)  

    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])

    angle = np.abs(radians*180.0/np.pi)     # transform rad to degrees
    
    return angle


def get_joints_coords(landmarks, mp_pose, image_height, image_width):

    joints_dict = {
        "mouth": [
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value].y * image_height
        ],
        "nose": [
            landmarks[mp_pose.PoseLandmark.NOSE.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.NOSE.value].y * image_height
        ],
        "leftEar": [
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * image_height
        ],
        "rightEar": [
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * image_height
        ],
        "leftShoulder": [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * image_height
        ],
        "rightShoulder": [
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * image_height
        ],
        "leftElbow": [
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * image_height
        ],
        "rightElbow": [
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * image_height
        ],
        "leftWrist": [
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * image_height
        ],
        "rightWrist": [
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * image_height
        ],
        "leftHip": [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
        ],
        "rightHip": [
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
        ],
        "leftKnee": [
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * image_height
        ],
        "rightKnee": [
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * image_height
        ],
        "leftAnkle": [
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * image_height
        ],
        "rightAnkle": [
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * image_width,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * image_height
        ]
    }
    
    return joints_dict

# visualize degrees
def visualize_degrees(location, image, angle):

    cv2.putText(
        image, str(int(angle)),
        (int(location[0]), int(location[1])),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA
    )

def main():

    # camera setup deppending the OS
    video = None
    if platform.system() == 'Linux':
        video = cv2.VideoCapture(0)                     # Default behavior for Linux
    elif platform.system() == 'Windows':
        video = cv2.VideoCapture(0, cv2.CAP_DSHOW)      # Default behavior for Windows
    else:
        raise Exception("Unsupported operating system")  
    

    # set window and resize
    cv2.namedWindow('Assistant', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Assistant', 1600, 900)              # window size

    # get the points
    mp_drawing = mp.solutions.drawing_utils     # drawing utils
    mp_pose = mp.solutions.pose                 # pose detection


    # define exercises data
    exercises = ['Chin tucks', 'Head shakes', 'Knees raises', 'Leg swings', 'Jumping jacks', 'Vertical jump']
    series =    [1, 1, 1, 1, 1, 1, 1, 1]
    reps =      [2000, 2, 2, 2, 2, 2, 2, 2]

    # iterate exercises to check all exercises
    for index, exercise in enumerate(exercises):
        rep_counter = serie_counter = 0       
        stage = 'Move NOW'           

        # start pose in exercise context
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while video.isOpened() and serie_counter < series[index]:
                ret, frame = video.read()
                if not ret:
                    print("No frame detected, closing app...")
                    break

                # use RBG colors
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # detect pose
                results = pose.process(image)

                # use BGR color
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                
                # get landmarks EXERCISE DETECTION AND FUNCTION USAGE LOGIC
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    image_height, image_width, _ = image.shape
                    
                    # get exercises's name
                    exercise_name = exercises[index]

                    # get all the coords to a dictionary
                    joint_coords = get_joints_coords(landmarks, mp_pose, image_height, image_width)

                    # get the function to the actual exercise
                    exercise_function = getattr(__import__('exercises_functions'), exercise_name.lower().replace(' ', ''))

                    # create a dict for the params and fill it, that's dinamically
                    params = {}

                    # get max reps and series
                    max_reps = reps[index]
                    max_series = series[index] 



                    if exercise_name == 'Chin tucks':
                        params['nose_angle'] = calculate_angle(joint_coords['leftShoulder'], joint_coords['rightShoulder'], joint_coords['nose'])  
                        visualize_degrees(joint_coords['nose'], image, params['nose_angle'])

                    elif exercise_name == 'Head shakes':                # use the 'X' position only
                        params['nose_position'] = joint_coords['nose'][0]     
                        params['left_shoulder_position'] = joint_coords['leftShoulder'][0] 
                        params['right_shoulder_position'] = joint_coords['rightShoulder'][0]

                    elif exercise_name == 'Jumping jacks':
                        params['shoulder_angle'] = calculate_angle(joint_coords['rightShoulder'], joint_coords['leftShoulder'], joint_coords['leftElbow'])
                        visualize_degrees(joint_coords['leftShoulder'], image, params['shoulder_angle'])

                        params['left_hip_position'] = joint_coords['leftHip'][0]
                        params['right_hip_position'] = joint_coords['rightHip'][0]


                        params['left_ankle_position'] = joint_coords['leftAnkle'][0]
                        params['right_angkle_position'] = joint_coords['rightAnkle'][0]

                    elif exercise_name == 'Leg swings':
                        params['knee_angle'] = calculate_angle(joint_coords['leftHip'], joint_coords['leftKnee'], joint_coords['leftAnkle'])
                        visualize_degrees(joint_coords['leftKnee'], image, params['knee_angle'])

                    elif exercise_name == 'Knees raises':
                        params['left_knee_position'] = joint_coords['leftKnee'][1]
                        params['left_hip_position'] = joint_coords['leftHip'][1]

                    elif exercise_name == 'Vertical jump':
                        # make the points with the image height and width
                        image_left_corner = [0, image_height]
                        image_right_corner = [image_width, image_height]

                        params['angle'] = calculate_angle(image_right_corner, image_left_corner, joint_coords['leftKnee'])
                        visualize_degrees(joint_coords['leftKnee'], image, params['angle'])
                        


                    # Call the exercise function with dynamic parameters
                    stage, rep_counter, serie_counter = exercise_function(
                        **params,   # Unpack parameters for the function
                        stage=stage,
                        rep_counter=rep_counter,
                        serie_counter=serie_counter,
                        max_reps=reps[index],
                        max_series=series[index]
                    )
                    

                # renderize detections
                if results.pose_landmarks is not None:
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )

                    # background rectangle for texts
                    cv2.rectangle(image, (0, 0), (150, image.shape[0]), (245, 117, 16), -1)
                    # set texts
                    cv2.putText(image, 'EXERCISE', (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'{exercise_name}', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'{index+1}/{len(exercises)}', (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'REPS', (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'{rep_counter}/{max_reps}', (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'SERIES', (15, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'{serie_counter}/{max_series}', (15, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, 'STAGE', (15, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f'{stage}', (15, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                
                # video size
                image = cv2.resize(image, (desired_width, desired_height))
                # show video
                cv2.imshow('Assistant', image)
                    
                # exit with 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    video.release()
                    cv2.destroyAllWindows()
                    return
            print(f"{exercise} finished!")
            time.sleep(0.250)


    video.release()
    cv2.destroyAllWindows()

    print(f"trainning finished!")

if __name__ == "__main__":
    main()
