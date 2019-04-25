import tensorflow as tf
import numpy as np
data1 = np.loadtxt("trainresult.txt",dtype=int)
new=data1.reshape(-1,1) 
x=np.where(new==69,21,new)  
trainlabel=np.zeros((len(data1),21))
for i in range(len(data1)):
    trainlabel[i,x[i]-1]=1
print(trainlabel.shape)

data2 = np.loadtxt("testresult.txt",dtype=int)
new2=data2.reshape(-1,1) 
x2=np.where(new2==69,21,new2)  
testlabel=np.zeros((len(data2),21))
for i in range(len(data2)):
    testlabel[i,x[i]-1]=1
print(testlabel.shape)
# name02='test_result.txt'
# np.savetxt(name02,testlabel,fmt='%d',newline='\n',delimiter='')
# name03='changed.txt'
# np.savetxt(name03,x,fmt='%d',newline='\n',delimiter='')

