import cv2


cap = cv2.VideoCapture(1)

while 1:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('is', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
