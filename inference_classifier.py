import pickle
import cv2
import mediapipe as mp
import numpy as np
import threading

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

labels_dict = {
    #no j or z right now
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K',
    10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P',
    15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U',
    20: 'V', 21: 'W', 22: 'X', 23: 'Y',
    #numbers 0 - 9
    24: '0', 25: '1', 26: '2', 27: '3', 28: '4',
    29: '5', 30: '6', 31: '7', 32: '8', 33: '9',
}

# Shared result storage — callback writes here, main loop reads from here
latest_result = None
result_lock = threading.Lock()
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def handle_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    with result_lock:
        latest_result = result

def draw_connections(frame, hand_landmarks, W, H):
    for connection in HAND_CONNECTIONS:
        start = connection.start
        end = connection.end
        x_start = int(hand_landmarks[start].x * W)
        y_start = int(hand_landmarks[start].y * H)
        x_end = int(hand_landmarks[end].x * W)
        y_end = int(hand_landmarks[end].y * H)
        cv2.line(frame, (x_start, y_start), (x_end, y_end), (255, 255, 255), 2)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    result_callback=handle_result
)

cap = cv2.VideoCapture(0)
timestamp = 0

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        H, W, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # detect_async doesn't block — result comes back via handle_result callback
        landmarker.detect_async(mp_image, timestamp)
        timestamp += 1

        with result_lock:
            result = latest_result

        if result and result.hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in result.hand_landmarks:
                for lm in hand_landmarks:
                    x_.append(lm.x)
                    y_.append(lm.y)
                    # Draw landmark dots
                    cx, cy = int(lm.x * W), int(lm.y * H)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                draw_connections(frame, hand_landmarks, W, H)

                for lm in hand_landmarks:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()