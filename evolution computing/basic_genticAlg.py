#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

DNA_LENGTH=10
POPULATION=10
CROSS_RATE=0.8
MUATION_RATE=0.03
GENERATIONS=200
X_BOUND=[0,80]
# print np.random.randint(0,POPULATION,size=10)
# print  np.random.randint(0,2,size=10).astype(np.bool)
def f(x):
    # return np.sin(10*x)+np.cos(2*x)*x
    return -x**2

def fitness(pred):
    # fit= np.e**pred
    # print pred
    # print  "fitness",fit
    return pred-np.min(pred)


def translateDNA(pol):
    '''
     2** np.arange(10)[::-1]
     [512 256 128  64  32  16   8   4   2   1] 二进制乘法 内积快速实现
    :param pol:
    :return:
    '''
    print pol
    binary_repres= 2** np.arange(DNA_LENGTH)[::-1]
    binary_dna=pop.dot(binary_repres.T)
    print "BINARY DATA:",binary_dna
   # float(2**DNA_SIZE-1) * X_BOUND[1]
    real_num=binary_dna/float(2**DNA_LENGTH-1)*X_BOUND[1]+X_BOUND[0]
    # a=np.Tsum(binary_dna,axis=0)
    print "REALVAL",real_num
    return real_num

def select(pop,fitness):
    index=np.random.choice(np.arange(POPULATION),POPULATION,p=fitness/fitness.sum())
    # print index
    return pop[index]
def crossover(parent,pop):
    if np.random.rand()<CROSS_RATE:
        i=np.random.randint(0,POPULATION,size=1)#choosing another paernt
        #下面两句还不太懂
        cross_points = np.random.randint(0,DNA_LENGTH,size=1) # 选择交叉点，这里假设每个基因都可以交叉
        # print "cross point",cross_points[0]
        # cross_points = np.random.randint(0,2,size=DNA_LENGTH).astype(np.bool) # 选择交叉点，这里假设每个基因都可以交叉
        # parent[cross_points[0]:] = pop[i][cross_points:] # 完成交叉
        another_parent=pop[i][0]
        # print "another :",another_parent
        # print "parent  :",parent
        parent[range(cross_points,10)]=another_parent[range(cross_points,10)]
        # print "exchange:",parent
        # print parent[range(0,3)]
    return parent
def muate(child):


    for point in range(DNA_LENGTH):
        if np.random.rand() < MUATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child


# print 2** np.arange(10)[::-1]
pop=np.random.randint(0,2,(POPULATION,DNA_LENGTH))

plt.ion()
x=np.linspace(X_BOUND[0],X_BOUND[1],2000)
plt.plot(x,f(x))
for _ in xrange(GENERATIONS):
    traslate_DNA=translateDNA(pop)
    F_values=f(traslate_DNA)
     # print traslate_DNA
    # something about plotting
    if 'sca' in globals(): sca.remove()
    print "translate values",traslate_DNA
    sca = plt.scatter(traslate_DNA, F_values, s=200, lw=0, c='red', alpha=0.5); plt.pause(0.1)

    fit_ness=fitness(F_values)
    pop=select(pop,fit_ness)   #根据适应度，选出另一批个体

    pop_copy=pop.copy()
    for parent in pop:
        child=crossover(parent,pop_copy)
        child=muate(child)
        parent[:]=child

    print("Most fitted DNA: ", traslate_DNA[np.argmax(fit_ness)],"max value:",f(traslate_DNA[np.argmax(fit_ness)]))
plt.ioff()
plt.show()