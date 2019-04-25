import tensorflow as tf
import numpy as np
import ReadImage

#read data
traindata =ReadImage.train_data
trainresult=ReadImage.train_label
testdata= ReadImage.test_data
testresult=ReadImage.test_label

#define a function to calculate accuracy
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys: v_ys})
    return result

def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

#define a 2 dimensional  convolutional neural layer
def conv2d(x,W):
    #strides[buntch=1,x_movement.y_movement,in_channels=1]
    #conv2d(input,filter,strides,pading(SAME,VAILD),name)
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#pooling 
def max_pool_2x2(x):
    #ksize means windows size
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1024])
ys = tf.placeholder(tf.float32, [None, 21])
#reshape(1024=>32x32)
x_image=tf.reshape(xs,[-1,32,32,1])

#conv1 layer
#patch is 5x5,input feature is 1,output(feature in this layer) is 20
W_conv1=weight_variable([5,5,1,20])
b_conv1=bias_variable([20])
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size=32x32x20
h_pool1=max_pool_2x2(h_conv1)#output size =16x16x20

#conv2 layer
W_conv2=weight_variable([5,5,20,50])
b_conv2=bias_variable([50])
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size=16x16x50
h_pool2=max_pool_2x2(h_conv2)#output size =8x8x50

#func1 layer
W_fc1=weight_variable([8*8*50,500])
b_fc1=bias_variable([500])
h_pool2_flat=tf.reshape(h_pool2,[-1,8*8*50])
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop=tf.nn.dropout(h_fc1,0.4)

#func2 layer
W_fc2=weight_variable([500,21])
b_fc2=bias_variable([21])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction,1e-10,1.0)), reduction_indices=[1]))

#train 
xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=prediction, labels=ys), name="xent")

#Use Adam Optimizer to train,learning rate is 4e-5
train_step = tf.train.AdamOptimizer(4e-5).minimize(xent)
#
#Code for creating a new batch
#
epochs_completed = 0
index_in_epoch = 0
num_examples = traindata.shape[0]

# serve data by batches
def next_batch():

    global traindata
    global trainresult
    global index_in_epoch
    global epochs_completed

    start = index_in_epoch
    index_in_epoch +=100

    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        traindata = traindata[perm]
        trainresult = trainresult[perm]
        # start next epoch
        start = 0
        index_in_epoch = 100
        assert 100 <= num_examples
    end = index_in_epoch
    #print(train_images[start:end])
    return traindata[start:end], trainresult[start:end]

#Initializition
sess = tf.Session()
init=tf.initialize_all_variables()
sess.run(init)
#interations for 1000 steps
for i in range(500):
    # use minibatch dataset to train, each batch contains 100 data
    batch_X, batch_Y = next_batch()
    sess.run(train_step, feed_dict={xs: batch_X, ys: batch_Y})
#     sess.run(train_step, feed_dict={xs: traindata, ys: trainresult})
    if i % 1 == 0:
        #show the train result
      print(i)
      print(compute_accuracy(traindata,trainresult))
      print(compute_accuracy(testdata,testresult))
print(compute_accuracy(traindata,trainresult))

print(compute_accuracy(testdata,testresult))
