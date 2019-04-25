import numpy as np
traindata = np.loadtxt("traindata.txt")
# traindata = tf.convert_to_tensor(traind)
print(traindata.shape)
trainresult = np.loadtxt("train_result.txt")
tresult=trainresult.reshape(-1,21)
print(tresult)