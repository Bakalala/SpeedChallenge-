import cv2 as cv
import numpy as np

height = 640
width = 330
layers = 3

#Steps to accomplish
#Save images (as individual files, maybe even continiosuly ?)
#process individual images
#get relathionship between 2 images (given the speed)
#put information in neural net
#validate network
#test network




# Opens the Video file
cap = cv.VideoCapture('speed_challenge_2017/data/train.mp4')
i = 1
path = 'frames/'

ret, frame = cap.read()
prvs = frame[20:350, 0:640]
prvs = cv.cvtColor(prvs,cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame[20:350, 0:640])
hsv[...,1] = 255

# Stores images in file
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break


    next = frame[20:350, 0:640]
    next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


    #cv.imwrite(path + 'test' + str(i) + '.jpg', frame)
    cv.imwrite(path + 'opticalhsv' + str(i) + '.png',bgr)

    i += 1
    if i%100 == 0 :
        print(i)

    prev = next

cap.release()
cv.destroyAllWindows()

print(i)