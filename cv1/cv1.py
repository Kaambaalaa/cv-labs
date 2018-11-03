import glob
import cv2


image_list = []
ethalonImg = 'input\IMG_0660.JPG'
for filename in glob.glob('input/*.jpg'):
    if(ethalonImg not in filename):
        image_list.append(filename)

def kaze_match(im1_path, im2_path, f):
    # load the image and convert it to grayscale
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)
    # if((descs1 not NoneType) and (descs2 not 'NoneType')):
    try:
        print("img: {}, keypoints: {}, descriptors: {}".format(im1_path, len(kps1), descs1.shape))
        print("img: {}, keypoints: {}, descriptors: {}".format(im2_path, len(kps2), descs2.shape))

    # Match the features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
        good = []
        loc = 0;
        for m,n in matches:
            if m.distance < 0.85*n.distance:
                good.append([m])
                loc += n.distance - m.distance
        info = "mathes: {}, good: {}, delta: {}, localization fault: {}".format(len(matches), len(good), len(matches) - len(good), loc / len(good))
        f.write(info + '\n')
        print(info)

        im3 = cv2.drawMatchesKnn(gray1, kps1, gray2, kps2, good, None, flags=2)

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        imS = cv2.resize(im3, (960, 540))  # Resize image
        cv2.imshow("output", imS)
    except Exception:
        print("error")


# відносна кількість правильно суміщених ознак
# похибка локалізації (відстань між реальним розміщенням предмета в кадрі та розпізнаним)
# та відносний час обробки фото в залежності від розміру зображення.
# Метрики мають зберегтись у файлику для подальших досліджень.

f = open("result.txt", "a")
for file in image_list:
    kaze_match(ethalonImg, file, f)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
f.close()
cv2.destroyAllWindows()