import model
import cv2 as cv
import numpy as np



height = 640
width = 330
layers = 3

max =  20399

speed = np.loadtxt('speed_challenge_2017/data/train.txt')

path = 'frames/'

i = 1

img = cv.imread(path + 'opticalhsv' + str(i) + '.png')

# print(img)

train = []

for i in range (1,max):
    img = cv.imread(path + 'opticalhsv' + str(i) + '.png')
    train.append(img)
    if i%100 == 0 :
        print(i)


x_input = np.array([train])
x_input = x_input.reshape(-1,height,width,layers)
print(x_input.shape)

#np.save('input', x_input)

# trainx = np.load('input')

# input shape
# frame = frame.reshape(-1,height,width,layers)

model = model.neuralNet()

model.load_weights('model.h5')

history = model.fit(x_input, speed[1:max], epochs = 5, validation_split=0.3)

model.save("model.h5")

test = x_input[1].reshape(-1,height,width,layers)

print(model.predict(test))