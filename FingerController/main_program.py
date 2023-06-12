import cv2
import mediapipe as mp
import numpy as np
import torch
from net import SimpleConvLSTM2
import pyautogui

def pause():
    pyautogui.press('space')

def play():
    pyautogui.press('space')

def move_10():
    pyautogui.press('right')

def move_m10():
    pyautogui.press('left')

def screen_max():
    pyautogui.press('f')

def screen_min():
    pyautogui.press('f')

def volume_up():
    pyautogui.press('up')

def volume_down():
    pyautogui.press('down')

def page_close():
    pyautogui.hotkey('alt', 'f4')

actions = ['play', 'pause', 'max_size', 'min_size', 'vol_up', 'vol_down', 'close_screen','video_forward_10s','video_backward_10s']
seq_length = 30
print("hello")
model = SimpleConvLSTM2()
model.load_state_dict(torch.load('./test9_6_new.pth',map_location=torch.device('cpu')))
model.eval()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    img0 = img.copy()

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for hand_landmarks in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))
            for j, lm in enumerate(hand_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
            v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
            v = v2 - v1
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n', v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                        v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
            angle = np.degrees(angle)

            angle_label = np.array([angle], dtype=np.float32)

            palm_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
                                    hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z])
            thumb_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                     hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                                     hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z])
            index_finger_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z])
            middle_finger_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
                                             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z])
            ring_finger_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
                                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z])
            pinky_finger_vector = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y,
                                            hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z])

            palm_thumb_vector = thumb_vector - palm_vector
            palm_index_finger_vector = index_finger_vector - palm_vector
            palm_middle_finger_vector = middle_finger_vector - palm_vector
            palm_ring_finger_vector = ring_finger_vector - palm_vector
            palm_pinky_finger_vector = pinky_finger_vector - palm_vector

            palm_thumb_distance = np.linalg.norm(palm_thumb_vector)
            palm_index_finger_distance = np.linalg.norm(palm_index_finger_vector)
            palm_middle_finger_distance = np.linalg.norm(palm_middle_finger_vector)
            palm_ring_finger_distance = np.linalg.norm(palm_ring_finger_vector)
            palm_pinky_finger_distance = np.linalg.norm(palm_pinky_finger_vector)

            palm_vectors = [palm_thumb_vector, palm_index_finger_vector, palm_middle_finger_vector,
                            palm_ring_finger_vector, palm_pinky_finger_vector]
            palm_distances = [palm_thumb_distance, palm_index_finger_distance, palm_middle_finger_distance,
                              palm_ring_finger_distance, palm_pinky_finger_distance]

            d = np.concatenate(
                [joint.flatten(), np.array(palm_vectors).flatten(), np.array(palm_distances),angle])

            seq.append(d)

            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

            input_data = np.concatenate([input_data, np.zeros((1, 30, 2))], axis=2)

            input_data = input_data.reshape((1, 30, 11, 11))

            y_pred = model(torch.Tensor(input_data))
            conf, index = torch.max(y_pred, 1)

            if conf < 0.7:
                continue

            action = actions[index]
            action_seq.append(action)

            if len(action_seq) < 9:
                continue

            this_action = '?'

            if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                this_action = action
                if last_action != this_action:

                    if this_action == 'play':
                        play()
                    elif this_action == 'pause':
                        pause()
                    elif this_action == 'max_size':
                        screen_max()
                    elif this_action == 'min_size':
                        screen_min()
                    elif this_action == 'vol_up':
                        volume_up()
                    elif this_action == 'vol_down':
                        volume_down()
                    elif this_action == 'close_screen':
                        page_close()
                    elif this_action == 'video_forward_10s':
                        move_10()
                    elif this_action == 'video_backward_10s':
                        move_m10()
                    last_action = this_action
                else:
                    if this_action == 'vol_up':
                        volume_up()
                    elif this_action == 'vol_down':
                        volume_down()
                    elif this_action == 'video_forward_10s':
                        move_10()
                    elif this_action == 'video_backward_10s':
                        move_m10()

            cv2.putText(img, f'{this_action.upper()}', org=(int(hand_landmarks.landmark[0].x * img.shape[1]), int(hand_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
