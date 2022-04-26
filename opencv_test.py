import cv2


if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        if ret == 1:
            cv2.imshow("test camera", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break #中断循环
    cap.release()
    cv2.destroyAllWindows()