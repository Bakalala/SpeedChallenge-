import model
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




# height = 640
# width = 330
# layers = 3

height = 220
width = 66
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
speed = speed[1:max]

randomize = np.arange(len(x_input))
np.random.shuffle(randomize)
x_input = x_input[randomize]
speed = speed[randomize]

model = model.neuralNet()

#model.load_weights('model.h5')

history = model.fit(x_input, speed, epochs = 20
                    , validation_split=0.2, shuffle=True)



model.save("model.h5")

print(history.history.keys())

# # Plot training & validation accuracy values
# plt.plot(history.history['mse'])
# plt.plot(history.history['val_mse'])
# plt.title('Model accuracy')
# plt.ylabel('MSE')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()



test = x_input[1].reshape(-1,height,width,layers)

print(model.predict(test))
print(speed[1])
