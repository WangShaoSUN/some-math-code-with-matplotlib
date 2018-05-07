#encoding=utf-8
import numpy as np
h=20  #hidden layer size
N=100 #number of points per class
D=2   #dimensionally
K=3   #number of classes

X=np.zeros((N*K,D))
y=np.zeros((N*K,K),dtype='uint8') #class label
# print y.shape
for i in xrange(3):
    index=range(N*i,N*(i+1))
    r=np.linspace(0.0,1,N)
    t=np.random.randn(N)*0.2+ np.linspace(i*4,(i+1)*4,N)
    X[index] = np.c_[r*np.sin(t), r*np.cos(t)]
y[range(0,100)]=[1,0,0]
y[range(100,200)]=[0,1,0]
y[range(200,300)]=[0,0,1]
# print y[[0,100,200],[0,1,2]]
# print y

import matplotlib.pyplot as plt

# plt.scatter(X[:,0],X[:,1],c=y,s=40)
# plt.show()

W1=np.random.random((D,h))
b1=np.zeros((1,h))
W2=np.random.random((h,K))
b2=np.zeros((1,K))

num_number=X.shape[0]
# print num_number
def sigmoid(x):
    return 1/(1+np.exp(-x))
# print np.array([1,2,3])*np.array([2,3,5])
# print sigmoid(12)
for i in xrange(100000):

    #use relu activation function
    hidden_layer=sigmoid(np.dot(X,W1))
    # print "a1",hidden_layer.shape

    derive=hidden_layer*(1-hidden_layer)
    # print derive.shape


    a2=np.dot(hidden_layer, W2)+b2
    # print "OUTPUT",a2.shape
    scores=sigmoid(a2)
    # print" scores",scores
    y_hat=scores-y
    loss=np.sum(y_hat*y_hat)
    # print loss
    # print y_hat

    if i%1000==0:
        print "iteration %d:loss %f"%(i,loss)
    L2_delta=y_hat*(scores*(1-scores))
    # print L2_delta.shape
    dW2=np.dot(hidden_layer.T,L2_delta)
    # print dW2
    db2=np.sum(y_hat,axis=0)

    dhidden=np.dot(L2_delta,W2.T)*derive
    # print dhidden.shape

    dw=np.dot(X.T,dhidden)
    # print dw

    W1-=0.1*dw
    W2-=0.01*dW2
# print   scores
predict=np.argmax(scores,axis=1)
print predict
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z =(np.dot(sigmoid( np.dot(np.c_[xx.ravel(), yy.ravel()],W1)),W2) )
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_net.png')
plt.show()

