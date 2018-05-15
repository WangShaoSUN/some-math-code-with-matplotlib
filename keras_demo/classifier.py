#encoding=utf-8
import cv2
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.optimizers import RMSprop
from keras import losses
import numpy as np
import matplotlib.pyplot as plt
# from keras.activations import .

(X_train,y_train),(X_test,y_test)=mnist.load_data()
print y_train
# plt.imshow(X_train[0], interpolation = "none", cmap = "Greys")
# plt.colorbar()
# plt.show()


''' data preprocessing '''
X_train=X_train.reshape(X_train.shape[0],-1)
print type(X_train)
x=np.expand_dims(X_train[0],axis=0)
print x.shape

# print np.max(y_test[0],axis=0)

X_test=X_test.reshape(X_test.shape[0],-1)

y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)
# print y_test
# print X_train.shape   #将图片拉成列向量
print y_train[0].argmax(axis=0)
# print  a
#to build your neural net

model=Sequential([
    Dense(32,input_dim=784),
    Activation('sigmoid'),

    Dense(10),
    Activation('softmax'),
])

rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

model.compile(
    optimizer=rmsprop,
    loss=losses.categorical_crossentropy,
    metrics=['accuracy'],
)

model.fit(X_train,y_train,nb_epoch=2,batch_size=100)

loss,accuracy=model.evaluate(X_test,y_test)

print "loss",loss
print "accuracy",accuracy
# x=np.expand_dims(X_train[0],axis=0)
# z= model.predict(x)
# print z[0]
# print z[0].argmax(axis=0)