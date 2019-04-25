from PIL import Image
import os.path
import numpy as np
import tensorflow as tf

imgTrainData=[]
imgTrainLabels=[]
imgTestData=[]
imgTestLabels=[]
for i in range(12,15):
    List=os.listdir('PIE/'+str(i))  
    for j in range(1,len(List)+1):
        fr=Image.open('PIE/'+str(i)+'/'+str(j)+'.jpg')
        im_array = np.array(fr).reshape((-1,1024))
        trainsize=round(len(List)*0.7)
        if(j<trainsize):
           imgTrainData.append(im_array)
           imgTrainLabels.append(i)
        else:
           imgTestData.append(im_array)
           imgTestLabels.append(i)

test=np.array(imgTestData)
test_data=np.array(test).reshape((-1,1024))
test1=np.array(imgTestLabels)
test_label=np.array(test1).reshape((-1,1))
test3=np.array(imgTrainData)
train_data=np.array(test3).reshape((-1,1024))
test4=np.array(imgTrainLabels)
train_label=np.array(test4).reshape((-1,1))

trainlabel=np.zeros((len(train_label),21))

for i in range(len(train_label)):
    trainlabel[i,train_label[i]-1]=1
print(trainlabel.shape)

 
testlabel=np.zeros((len(test_label),21))
for i in range(len(test_label)):
    testlabel[i,test_label[i]-1]=1
print(testlabel.shape)



