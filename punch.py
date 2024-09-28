import cv2
import mediapipe as mp
import numpy as np
import time
import threading

buffer_elapsed = False

def timer_buffer(seconds):
    def timer():
        global buffer_elapsed 
        buffer_elapsed = False
        time.sleep(seconds)
        buffer_elapsed = True
    
    timer_thread = threading.Thread(target=timer)
    timer_thread.start()
    

punch_stance_sequence = []
punch_stance_angle_sequence = []
punch_sequence = []
punch_angle_sequence = []
timer_buffer(5)

def detect_pose_live():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pass_count = 1
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        # read frame
        _, frame = cap.read()
        try:
            # convert to RGB
            print(buffer_elapsed)
            if pass_count < 6:
                # timer = time.time()
                if practice_loop(frame, pose, pass_count) and buffer_elapsed:
                    pass_count += 1
                    timer_buffer(5)
                # pass_count = practice_loop(frame, pose, mp_pose, mp_drawing, pass_count)
                cv2.putText(frame, str(pass_count-1), (200, 200), cv2.FONT_HERSHEY_COMPLEX,4, (255,255,255),2, cv2.LINE_8)
                
                # draw skeleton on the frame
                mp_drawing.draw_landmarks(frame, pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # display the frame
            
            # cv2.putText(frame, "yippee", (200, 200), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,0),2, cv2.LINE_8)
            cv2.imshow('Output', frame)
            
        except Exception as e:
            print(f"Error: {e}")
            break
            
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def compare_angle_sequences(live, standing):
    if not live or not standing:
        return 0.0
    # print(live)
    accuracy = 0
    # for i in range(0, len(live), 8):
    #     accuracy = (1 - (compute_mse(standing,live[len(live)-8:len(live)]))/(180**2))*100

    if(len(live)>0):
        if(len(live)%8 == 0):
            accuracy = (1 - (compute_mse(standing,live[len(live)-8:len(live)]))/(90**2))*100
    return accuracy


def detect_punch_stance(image_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image = cv2.imread(image_path)
    # convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process the image for pose detection
    pose_results = pose.process(image_rgb)

    relevant_results = pose_results.pose_landmarks.landmark
    for lndmrk in relevant_results:
        x = lndmrk.x
        y = lndmrk.y
        z = lndmrk.z
        punch_stance_sequence.append([x, y, z])
    find_angle_sequence(punch_stance_sequence, punch_stance_angle_sequence)
    # draw skeleton on the image
    mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # cv2.imshow('Output', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
def detect_punch(image_path):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    image = cv2.imread(image_path)
    # convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # process the image for pose detection
    pose_results = pose.process(image_rgb)

    relevant_results = pose_results.pose_landmarks.landmark
    for lndmrk in relevant_results:
        x = lndmrk.x
        y = lndmrk.y
        z = lndmrk.z
        punch_sequence.append([x, y, z])
    find_angle_sequence(punch_sequence, punch_angle_sequence)
    # draw skeleton on the image
    mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    # cv2.imshow('Output', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def find_angle_sequence(sequence, angle_sequence):
    mp_pose = mp.solutions.pose
    # print(sequence)
    right_forearm_1416 = compute_vector_difference(sequence[mp_pose.PoseLandmark.RIGHT_WRIST.value], sequence[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    right_upper_arm_1214 = compute_vector_difference(sequence[mp_pose.PoseLandmark.RIGHT_ELBOW.value], sequence[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    right_body_1224 = compute_vector_difference(sequence[mp_pose.PoseLandmark.RIGHT_ELBOW.value], sequence[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    right_thigh_2426 = compute_vector_difference(sequence[mp_pose.PoseLandmark.RIGHT_HIP.value], sequence[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    right_calf_2628 = compute_vector_difference(sequence[mp_pose.PoseLandmark.RIGHT_KNEE.value], sequence[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    left_upper_arm_1113 = compute_vector_difference(sequence[mp_pose.PoseLandmark.LEFT_SHOULDER.value], sequence[mp_pose.PoseLandmark.LEFT_ELBOW.value])
    left_body_1123 = compute_vector_difference(sequence[mp_pose.PoseLandmark.LEFT_SHOULDER.value], sequence[mp_pose.PoseLandmark.LEFT_HIP.value])
    left_forearm_1315 = compute_vector_difference(sequence[mp_pose.PoseLandmark.LEFT_ELBOW.value], sequence[mp_pose.PoseLandmark.LEFT_WRIST.value])
    left_thigh_2325 = compute_vector_difference(sequence[mp_pose.PoseLandmark.LEFT_HIP.value], sequence[mp_pose.PoseLandmark.LEFT_KNEE.value])
    left_calf_2527 = compute_vector_difference(sequence[mp_pose.PoseLandmark.LEFT_KNEE.value], sequence[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    right_elbow_angle = compute_angle(right_forearm_1416, right_upper_arm_1214)
    angle_sequence.append(right_elbow_angle)
    right_armpit_angle = compute_angle(right_upper_arm_1214, right_body_1224)
    angle_sequence.append(right_armpit_angle)
    right_hip_angle = compute_angle(right_body_1224, right_thigh_2426)
    angle_sequence.append(right_hip_angle)
    right_knee_angle = compute_angle(right_thigh_2426, right_calf_2628)
    angle_sequence.append(right_knee_angle)
    
    left_elbow_angle = compute_angle(left_forearm_1315, left_upper_arm_1113)
    angle_sequence.append(left_elbow_angle)
    left_armpit_angle = compute_angle(left_upper_arm_1113, left_body_1123)
    angle_sequence.append(left_armpit_angle)
    left_hip_angle = compute_angle(left_body_1123, left_thigh_2325)
    angle_sequence.append(left_hip_angle)
    left_knee_angle = compute_angle(left_thigh_2325, left_calf_2527)
    angle_sequence.append(left_knee_angle) 

def compute_vector_difference(final, initial):
    return [final[0] - initial[0],final[1]-initial[1], final[2]-initial[2]]
    
def compute_angle(vector1, vector2):
    angle = np.degrees(np.arccos(np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))))
    # print(angle)
    return angle

def compute_mse(ref_angles, live_angles):
    sum = 0

    for i in range(len(ref_angles)):
        dif = (ref_angles[i] - live_angles[i])**2
        if i is (0 or 1 or 4 or 5):
            dif  = dif*4/3
        else:
            dif = dif*0.75
        sum += dif
    return sum/8

def practice_loop(live_video, pose, pass_count): 
    if pass_count < 6:

        frame_rgb = cv2.cvtColor(live_video, cv2.COLOR_BGR2RGB)            
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            relevant_results = pose_results.pose_landmarks.landmark
            live_pose_sequence = []
            live_angle_sequence = []
            for lndmrk in relevant_results:
                if(lndmrk.visibility>0.7):
                    x = lndmrk.x
                    y = lndmrk.y
                    z = lndmrk.z
                else:
                    x = 0
                    y = 0
                    z = 0
                live_pose_sequence.append([x, y, z])
            find_angle_sequence(live_pose_sequence, live_angle_sequence)
            if live_angle_sequence:
                # Compare live_angle_sequence with punch_stance_angle_sequence
                accuracy = compare_angle_sequences(live_angle_sequence, punch_stance_angle_sequence)
                if not np.isnan(accuracy):
                    # print(f"Accuracy: {accuracy:.2f}%")
                    cv2.putText(live_video, f"{accuracy:.2f}%", (100, 100), cv2.FONT_HERSHEY_COMPLEX,4, (0,0,0),2, cv2.LINE_8)
                    # print(time.time()-timer)
                    if accuracy > 95:
                        cv2.putText(live_video, "now punch!", (200, 100), cv2.FONT_HERSHEY_COMPLEX,4, (0,0,0),2, cv2.LINE_8)
                        accuracy = compare_angle_sequences(live_angle_sequence, punch_angle_sequence)
                        cv2.putText(live_video, f"{accuracy:.2f}%", (100, 100), cv2.FONT_HERSHEY_COMPLEX,4, (0,0,0),2, cv2.LINE_8)
                        
                        if accuracy > 95:
                            pass_count += 1
                            return True
                
            # cv2.putText(live_video, str(pass_count), (200, 200), cv2.FONT_HERSHEY_COMPLEX,1, (255,0,0),2, cv2.LINE_8)
                    
            #     # draw skeleton on the frame
            # mp_drawing.draw_landmarks(live_video, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            # # display the frame
            # cv2.imshow('Output', live_video)
    return False
    


if __name__ == '__main__':
    # detect_pose_video('IMG_6999.mov')
    detect_punch_stance('punch_stance.jpg')
    detect_punch('punch.jpg')
    
    # print(len(standing_pose_sequence))
    # wait_time = 1
    # for i in range(3):
    #     time.sleep(wait_time)
    #     print(f"Waited for {wait_time * (i + 1)} seconds")
    detect_pose_live()

    