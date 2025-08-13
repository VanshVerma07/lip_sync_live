import cv2
import mediapipe as mp
import numpy as np
import os

# === Set output folder for saved metrics and landmarks ===
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

# Get the exact indices for the lips from MediaPipe's official FACEMESH_LIPS
LIPS_IDX = set()
for conn in mp_face_mesh.FACEMESH_LIPS:
    LIPS_IDX.update(conn)
LIPS_IDX = sorted(LIPS_IDX)  # Only plot these
LIPS_CONN = mp_face_mesh.FACEMESH_LIPS

def draw_lips(frame, face_landmarks):
    h, w = frame.shape[:2]
    # Draw white lip contour only
    mp_drawing.draw_landmarks(
        frame,
        face_landmarks,
        LIPS_CONN,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_style.get_default_face_mesh_contours_style()
    )
    # Draw green dots ONLY on lip landmarks
    for idx in LIPS_IDX:
        lm = face_landmarks.landmark[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
    return frame

cap = cv2.VideoCapture(0)
lip_metrics_list = []
lip_landmarks_list = []

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    print("Live lip landmarking started. Press Q to quit webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                frame = draw_lips(frame, face_landmarks)
                # Save per-frame lip metrics (width, height) and all lip landmarks
                mouth_points = np.array([
                    [face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]
                    for idx in LIPS_IDX
                ])
                # For width/height: using recommended indices (may need to adjust for best accuracy)
                # The leftmost/rightmost, top/bottom-most lips, or use indices:
                top = mouth_points[13]
                bottom = mouth_points[3]
                left = mouth_points[0]
                right = mouth_points[6]
                width = float(np.linalg.norm(left - right))
                height = float(np.linalg.norm(top - bottom))
                lip_metrics_list.append([width, height])
                lip_landmarks_list.append(mouth_points)
                cv2.putText(frame, f"Width:{width:.3f} Height:{height:.3f}", (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0),2)

        cv2.imshow("Live Lip Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# === Save all metrics and landmarks in output folder ===
lip_metrics_arr = np.array(lip_metrics_list)    # (frames, 2)
lip_landmarks_arr = np.array(lip_landmarks_list) # (frames, N, 2)

np.save(os.path.join(OUTPUT_DIR, "lip_metrics_live.npy"), lip_metrics_arr)
np.save(os.path.join(OUTPUT_DIR, "lip_landmarks_live.npy"), lip_landmarks_arr)

print(f"Lip metrics and landmarks saved in {OUTPUT_DIR}:")
print(f"- lip_metrics_live.npy (columns: width, height)")
print(f"- lip_landmarks_live.npy (shape: frames, N landmarks, x/y where N={len(LIPS_IDX)})")
