from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, MaxPooling2D, Convolution2D,Cropping2D
from scripts.reader import X_train, Y_train

model= Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.summary()
#training
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True, nb_epoch=7)

model.save('model.h5')