import cv2
import numpy as np
import argparse
import random as rng
WIDTH = 640
HEIGHT = 480
YELLOW = (0, 177, 177)
MAGENTA = (177, 0, 177)
ORANGE = (37, 117, 247)


def recordAndSaveVideo(cam, output):
    cap = cv2.VideoCapture(cam)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output, fourcc, 20.0, (WIDTH, HEIGHT))
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
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

# rng.seed(12345)
def drawContours(frame, contours, hierarchy):
    for i in range(len(contours)):
        color = YELLOW
        cv2.drawContours(frame, contours, i, color, 1, cv2.LINE_8, hierarchy, 0)
    return frame

def prepareRacoonContours(val):
    oriimg = cv2.imread('racoon.jpg')
    height, width, depth = oriimg.shape
    imgScaleWidth = WIDTH / (width * 2)
    imgScaleHeight = HEIGHT / (height )
    newX, newY = oriimg.shape[1] * imgScaleWidth, oriimg.shape[0] * imgScaleHeight
    newimg = cv2.resize(oriimg, (int(newX), int(newY)))
    threshold = val
    canny_output = cv2.Canny(cv2.cvtColor(newimg, cv2.COLOR_BGR2GRAY), threshold, threshold * 2)
    _, contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def backInBlack(fileName):
    racoonContours, hierarchy = prepareRacoonContours(50)
    cap = cv2.VideoCapture(fileName)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            color = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.line(color, (0, 0), (WIDTH, HEIGHT), ORANGE, 2)
            cv2.rectangle(color, (round(WIDTH * 4 / 8), round(HEIGHT * 4 / 8)), (round(WIDTH * 6 / 8), round(HEIGHT * 6 / 8)),
                          MAGENTA, 2)
            drawContours( color, racoonContours, hierarchy)
            cv2.imshow('frame', color)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


# demonstration
recordAndSaveVideo(0, 'output.avi')
backInBlack('output.avi')
