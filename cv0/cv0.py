import cv2

WIDTH = 640
HEIGHT = 480


def recordAndSaveVideo(cam, output):
    cap = cv2.VideoCapture(cam)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 20.0, (WIDTH, HEIGHT))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.line(frame, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), 3)
            cv2.rectangle(frame, (round(WIDTH / 4), round(HEIGHT / 4)), (round(WIDTH * 3 / 4), round(HEIGHT * 3 / 4)),
                          (0, 0, 0), 3)
            out.write(frame)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def backInBlack(fileName):
    cap = cv2.VideoCapture(fileName)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame', gray)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# demonstration
recordAndSaveVideo(1, 'output.avi')
backInBlack('output.avi')
