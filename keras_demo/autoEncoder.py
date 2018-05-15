#encoding=utf-8
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense,Input
from keras.models import Model
from keras import losses
(xtrain,_),(xtest,_)=mnist.load_data()

#normalization data
xtrain=xtrain.astype("float32")/255
xtest=xtrain.astype("float32")/255
# print xtrain.shape


xtrain=xtrain.reshape(60000,784)
xtest=xtest.reshape(len(xtest),np.prod(xtest.shape[1:]))

# print xtest.shape

encodingDim=32
InputImg=Input(shape=(784,))
hiddenLayer=Dense(encodingDim,activation='relu')(InputImg)
# hiddenLayer2=Dense(350,activation='relu')(hiddenLayer)

decode=Dense(784,activation='sigmoid')(hiddenLayer)

autoEncoder=Model(input=InputImg,output=decode)

#对压缩和解码重新定义两个神经网络
encoder=Model(input=InputImg,output=hiddenLayer)
encoderShaped=Input(shape=(encodingDim,))
decoderLayer=autoEncoder.layers[-1]
# print a
decoder=Model(input=encoderShaped,output=decoderLayer(encoderShaped))

#训练自编码器
autoEncoder.compile(optimizer='adam',loss=losses.binary_crossentropy)
autoEncoder.fit(xtrain,xtrain,epochs=50,batch_size=256,shuffle=True,validation_data=\
    (xtest,xtest))

encodeImg=encoder.predict(xtrain[0:20])
decodeImg=decoder.predict(encodeImg)

import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
n=10
for  i in range(n):

    #displaying original
    ax = plt.subplot(2,n,i+1)
    plt.imshow(xtrain[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #displaying predictions
    ax = plt.subplot(2,n,i+1+n)
    plt.imshow(decodeImg[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# from keras.layers import Input, Dense
# from keras.models import Model
# from keras.datasets import mnist
# import numpy as np
# import matplotlib.pyplot as plt
#
# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
# print(x_train.shape)
# print(x_test.shape)
#
# encoding_dim = 32
# input_img = Input(shape=(784,))
#
# encoded = Dense(encoding_dim, activation='relu')(input_img)
# decoded = Dense(784, activation='sigmoid')(encoded)
#
# autoencoder = Model(inputs=input_img, outputs=decoded)
# encoder = Model(inputs=input_img, outputs=encoded)
#
# encoded_input = Input(shape=(encoding_dim,))
# decoder_layer = autoencoder.layers[-1]
#
# decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))
#
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
# autoencoder.fit(x_train, x_train, epochs=12, batch_size=256,
#                 shuffle=True, validation_data=(x_test, x_test))
#
# encoded_imgs = encoder.predict(x_test[0:10])
# decoded_imgs = decoder.predict(encoded_imgs)
#
# n = 10  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()


