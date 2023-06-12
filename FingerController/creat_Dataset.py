import cv2
import mediapipe as mp
import numpy as np
import time
import os

actions = ['play', 'pause', 'max_size', 'min_size', 'vol_up', 'vol_down', 'close_screen','video_forward_10s','video_backward_10s']
secs_for_action = {'play': 60, 'pause':60, 'max_size': 60, 'min_size': 60, 'vol_up': 60, 'vol_down': 60, 'close_screen': 60, 'video_forward_10s':60,'video_backward_10s':60}

secs_rest = 5
seq_length = 30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

created_time = time.strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join('dataset', created_time)
os.makedirs(save_dir, exist_ok=True)

for i, action in enumerate(actions):
        data = []
        start_time = time.time()

        print(f'Collecting {action.upper()} action. Get ready...')

        while time.time() - start_time < secs_for_action[action]:
            ret, img = cap.read()
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

                results = hands.process(img_rgb)

                if results.multi_hand_landmarks is not None:
                    for hand_landmarks in results.multi_hand_landmarks:
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
                        angle_label = np.append(angle_label, i)

 
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


                        d = np.concatenate([joint.flatten(), np.array(palm_vectors).flatten(), np.array(palm_distances),angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            time_left = int(secs_for_action[action] - (time.time() - start_time))
            text = f'{action.upper()} action learning now. {time_left} seconds left'
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(f'{action}: {data.shape}')
        np.save(os.path.join(save_dir, f'raw_{action}'), data)

        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(f'{action} sequence: {full_seq_data.shape}')
        np.save(os.path.join(save_dir, f'seq_{action}'), full_seq_data)

        print(f'Finished collecting {action.upper()} action. Get ready for the next action...')

        if i < len(actions) - 1:
            next_action = actions[i + 1]
            print(f'Resting for {secs_rest} seconds...')
            time.sleep(secs_rest)
            time_left = secs_rest
            while time_left > 0:
                text = f'Waiting. {time_left} seconds left. Next Action is {next_action}'
                img_rest = np.zeros_like(img)
                cv2.putText(img_rest, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('img', img_rest)
                cv2.waitKey(1000)
                time_left -= 1

cap.release()
cv2.destroyAllWindows()
