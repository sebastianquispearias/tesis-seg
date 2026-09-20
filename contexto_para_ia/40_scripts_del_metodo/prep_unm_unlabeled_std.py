import cv2

video_path = r"G:\My Drive\UNM TalkBank Dysphagia\videos-selecionados\v025.avi"
cap = cv2.VideoCapture(video_path)
frame_idx = 3000  # empieza cerca del cambio
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("Video", frame)
    key = cv2.waitKey(0)
    if key == ord('d'):  # avanza
        frame_idx += 1
    elif key == ord('a'):  # retrocede
        frame_idx -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    elif key == ord('q'):  # salir
        break
    else:
        frame_idx += 1

cap.release()
cv2.destroyAllWindows()