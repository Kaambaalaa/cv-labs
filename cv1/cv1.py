import os, glob
import cv2
import time
import numpy as np
import math

image_list = []
ethalonImg = 'input\IMG_0660.JPG'
for filename in glob.glob('input/*.jpg'):
    if (ethalonImg not in filename):
        image_list.append(filename)


def matchCenter(img):
    hsv_min = np.array((0, 0, 153), np.uint8)
    hsv_max = np.array((255, 51, 255), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv, hsv_min, hsv_max)
    moments = cv2.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    x = int(dM10 / dArea)
    y = int(dM01 / dArea)
    return x, y


def kaze_match(kps1, descs1, im2_path, f):
    start_time = time.time()
    im2 = cv2.imread(im2_path)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    detector = cv2.AKAZE_create()
    (kps2, descs2) = detector.detectAndCompute(gray2, None)
    try:

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descs1, descs2, k=2)

        good = []
        center = [0, 0]
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
                center[0] += kps2[m.trainIdx].pt[0]
                center[1] += kps2[m.trainIdx].pt[1]
        matchTime = time.time() - start_time
        x, y = matchCenter(im2)
        # похибка локалізації (відстань між реальним розміщенням предмета в кадрі та розпізнаним)
        locError = math.sqrt((x - center[0] / len(good)) ** 2 + (y - center[1] / len(good)) ** 2)
        info = "img: {}, mathes: {}, good: {}, delta: {}, time/size: {} sec, localization error: {}" \
            .format(im2_path, len(matches), len(good), len(matches) - len(good),
                    matchTime / os.path.getsize(im2_path) * 1000000, locError)
        f.write(info + '\n')
        print(info)
        # im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2, good, None, flags=2)
        # cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        # imS = cv2.resize(im3, (960, 540))  # Resize image
        # cv2.imshow("output", imS)
    except Exception:
        print("error")

f = open("result.txt", "a")
im1 = cv2.imread(ethalonImg)
gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
detector = cv2.AKAZE_create()
(kps1, descs1) = detector.detectAndCompute(gray1, None)

for file in image_list:
    kaze_match(kps1, descs1, file, f)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
f.close()
cv2.destroyAllWindows()
