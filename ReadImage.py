from PIL import Image
import os.path
import numpy as np
import tensorflow as tf

#Read images to list,divide them into train data and test data
imgTrainData=[]
imgTrainLabels=[]
imgTestData=[]
imgTestLabels=[]
for i in range(13,15):
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

#convert data list into narray and change their shapes
test_data=np.array(imgTestData).reshape((-1,1024))
testlabel=np.array(imgTestLabels).reshape((-1,1))
train_data=np.array(imgTrainData).reshape((-1,1024))
trainlabel=np.array(imgTrainLabels).reshape((-1,1))

#convert train and test label  to 1*21(for example:1=>[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
train_label=np.zeros((len(trainlabel),21))
for i in range(len(trainlabel)):
    train_label[i,trainlabel[i]-1]=1
print(train_label.shape)

 
test_label=np.zeros((len(testlabel),21))
for i in range(len(testlabel)):
    test_label[i,testlabel[i]-1]=1
print(test_label.shape)



