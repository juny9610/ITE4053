import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tensorflow.python.ops.gen_array_ops import shape

training_epochs = 20

# Mini-batch
def batch_data(shuffled_idx, batch_size, data, labels, start_idx):
    idx = shuffled_idx[start_idx:start_idx+batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# CNN
def build_CNN_classifier(x):
    x_image = x

    # layer 1
    # 5*5 사이즈, 64출력
    # 3*3 pooling, 2칸씩 적용
    W1 = tf.get_variable(name="W1", shape=[5, 5, 3, 64], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name="b1", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
    c1 = tf.nn.conv2d(x_image, W1, strides=[1, 1, 1, 1], padding='SAME')
    l1 = tf.nn.relu(tf.nn.bias_add(c1, b1))
    l1 = tf.layers.batch_normalization(l1)
    l1_pool = tf.nn.max_pool(l1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer 2
    # 5*5 사이즈, 128출력
    # 3*3 pooling, 2칸씩 적용
    W2 = tf.get_variable(name="W2", shape=[5, 5, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name="b2", shape=[128], initializer=tf.contrib.layers.xavier_initializer())
    c2 = tf.nn.conv2d(l1_pool, W2, strides=[1, 1, 1, 1], padding='SAME')
    l2 = tf.nn.relu(tf.nn.bias_add(c2, b2))
    l2 = tf.layers.batch_normalization(l2)
    l2_pool = tf.nn.max_pool(l2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # layer 3
    # 5*5 사이즈, 256출력
    # 3*3 pooling, 2칸씩 적용
    W3 = tf.get_variable(name="W3", shape=[5, 5, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable(name="b3", shape=[256], initializer=tf.contrib.layers.xavier_initializer())
    c3 = tf.nn.conv2d(l2_pool, W3, strides=[1, 1, 1, 1], padding='SAME')
    l3 = tf.nn.relu(tf.nn.bias_add(c3, b3))
    l3 = tf.layers.batch_normalization(l3)
    l3_pool = tf.nn.max_pool(l3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    l3_flat = tf.reshape(l3_pool, [-1, 8*8*64])

    # Fully connected
    W_fc = tf.get_variable(name="W_fc", shape=[8*8*64, 10], initializer=tf.contrib.layers.xavier_initializer())
    b_fc = tf.get_variable(name="b_fc", shape=[10], initializer=tf.contrib.layers.xavier_initializer())
    
    # activation_function = softmax
    logits = tf.nn.bias_add(tf.matmul(l3_flat, W_fc), b_fc)
    hypothesis = tf.nn.softmax(logits)

    return hypothesis, logits

# CheckPoint
ckpt_path = "output/"

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")

# Input normalization (0-1)
x_train = x_train/x_train.max()

dev_num = len(x_train) // 4

x_dev = x_train[:dev_num]
y_dev = y_train[:dev_num]

x_train = x_train[dev_num:]
y_train = y_train[dev_num:]

y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_dev_one_hot = tf.squeeze(tf.one_hot(y_dev, 10),axis=1)

y_pred, logits = build_CNN_classifier(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

# Adam optimizer
train_step = tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam').minimize(cost)

batch_size = 128
total_batch = int(len(x_train)/batch_size) if len(x_train)%batch_size == 0 else int(len(x_train)/batch_size) + 1

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("학습시작")

    for epoch in range(training_epochs):
        start = 0
        avg_cost = 0
        shuffled_idx = np.arange(0, len(x_train))
        np.random.shuffle(shuffled_idx)

        for i in range(total_batch):
            batch = batch_data(shuffled_idx, batch_size, x_train, y_train_one_hot.eval(), i*batch_size)
            c, _ = sess.run([cost, train_step], feed_dict={x: batch[0], y: batch[1]})
            avg_cost += c/total_batch
        # Epoch, loss 값 확인
        print('Epoch: ', '%d/%d' %(epoch+1, training_epochs), 'Cost =', '{:.9f}'.format(avg_cost))
        
    saver = tf.train.Saver()
    saver.save(sess, ckpt_path)
    saver.restore(sess, ckpt_path)

    y_prediction = np.argmax(y_pred.eval(feed_dict={x: x_dev}), 1)
    y_true = np.argmax(y_dev_one_hot.eval(), 1)
    dev_f1 = f1_score(y_true, y_prediction, average="weighted") # f1 스코어 측정
    print("dev 데이터 f1 score: %f" % dev_f1)

    # 밑에는 건드리지 마세요
    x_test = np.load("data/x_test.npy")
    test_logits = y_pred.eval(feed_dict={x: x_test})
    np.save("result", test_logits)

