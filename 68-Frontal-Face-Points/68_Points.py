from imutils.video import VideoStream
from imutils import face_utils
import time
import dlib
import cv2


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(r_eyeb_Start, r_eyeb_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(l_eyeb_Start, l_eyeb_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(r_eye_Start, r_eye_End) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(l_eye_Start, l_eye_End) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
(jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

vs = VideoStream().start()
time.sleep(1.0)

while True:
    counter =0
    frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouth = shape[mStart:mEnd]
        reyeb = shape[r_eyeb_Start:r_eyeb_End]
        leyeb = shape[l_eyeb_Start:l_eyeb_End]
        reye = shape[r_eye_Start:r_eye_End]
        leye = shape[l_eye_Start:l_eye_End]
        nose = shape[nStart:nEnd]
        jaw = shape[jStart:jEnd]

        for (x,y) in mouth:
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)
        for (x,y) in reyeb:
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
        for (x,y) in leyeb:
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
        for (x,y) in reye:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        for (x,y) in leye:
            cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
        for (x,y) in nose:
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        for (x,y) in jaw:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
