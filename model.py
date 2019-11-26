from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Dense, Flatten
from keras.metrics import accuracy


# height = 640
# width = 330
# layers = 3

height = 220
width = 66
layers = 3

def neuralNet():

    model = Sequential()


    #normalize the image
    #bring rgb values of 0 - 255 to -1 and 1

    model.add(Lambda(lambda x: x/255, input_shape = (height, width, layers)))

    model.add(Conv2D(24, (5,5), strides = (2, 2)))

    model.add(Conv2D(36, (5,5), strides = (2, 2)))

    model.add(Conv2D(48, (5,5), strides = (2, 2)))

    model.add(Conv2D(64, (3,3)))

    model.add(Conv2D(64, (3,3)))

    model.add(Flatten())

    model.add(Dense(100))

    model.add(Dense(50))

    model.add(Dense(10))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



# model = neuralNet()
# test = np.zeros([1,height,width,layers])
# print(model.predict(test))
