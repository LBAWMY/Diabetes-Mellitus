import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops
import h5py
import matplotlib.pyplot as plt
import numpy as np

ops.reset_default_graph()
sess = tf.Session()

#all/half predict result (use only draft_params and params feature)
# xgb_train_preds = pd.read_csv('xgb_train_preds.csv')
# dart_train_preds = pd.read_csv('dart_train_preds.csv')
# rf_train_preds = pd.read_csv('rf_train_preds.csv')
# train_label = pd.read_csv('../data/feature/offline_train_feat.csv')

xgb_preds = pd.read_csv('xgb_preds.csv')
dart_preds = pd.read_csv('dart_preds.csv')
rf_preds = pd.read_csv('rf_preds.csv')
lightdm1_preds = pd.read_csv('lgdm_preds1.csv')
# test_label = pd.read_csv('../data/feature/offline_test_feat.csv')
test_label = pd.read_csv('../data/feature/all_test_feat.csv')


# w = [0.3,0.3,0.4] # acquire by tensorflow
# xgb_preds.score_all = w[0]*xgb_preds.score_all + w[1]*dart_preds.score_all + w[2]*rf_preds.score_all

# x_vals_train = np.append(xgb_preds.score_all.values.reshape(-1,1), dart_preds.score_all.values.reshape(-1,1), axis=1)
# x_vals_train = np.append(x_vals_train, rf_preds.score_all.values.reshape(-1,1), axis=1)
# x_vals_train = np.append(x_vals_train, lightdm1_preds.score_all.values.reshape(-1,1), axis=1)
x_vals_train = np.append(xgb_preds.score_all.values.reshape(-1,1), lightdm1_preds.score_all.values.reshape(-1,1), axis=1)
# x_vals_train = np.append(x_vals_train, rf_preds.score_all.values.reshape(-1,1), axis=1)
# x_vals_train = np.append(x_vals_train, lightdm1_preds.score_all.values.reshape(-1,1), axis=1)

y_vals_train = test_label['血糖'].values


learning_rate = tf.placeholder(dtype=tf.float32)
batch_size = 1000
iterations = 50000
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

A = tf.Variable(tf.random_normal(shape=[2, 1]))
A1 = tf.div(A,tf.reduce_sum(A))
# b = tf.Variable(tf.random_normal(shape=[1, 1]))
# model_output = tf.add(tf.matmul(x_data, A), b)
model_output = tf.matmul(x_data, A1)

loss = tf.div(tf.reduce_mean(tf.square(y_target - model_output)), 2.)
init = tf.global_variables_initializer()
sess.run(init)
my_opt = tf.train.GradientDescentOptimizer(learning_rate)
train_step = my_opt.minimize(loss)

loss_vec = []
train_loss_vec = []
test_loss_vec = []
for i in range(iterations):
    # rand_index = np.random.choice(x_vals_train.shape[0], size=batch_size)
    # rand_x = x_vals_train[rand_index]
    # rand_y = np.transpose([y_vals_train[rand_index]])
    rand_x = x_vals_train
    rand_y = np.transpose([y_vals_train])
    # Learning rate control
    rate = 0.008

    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y, learning_rate: rate})
    if (i+1)%500 == 0:
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss_vec.append(temp_loss)

        print('Step #' + str(i+1) + ': Loss = ' + str(temp_loss))
        print(sess.run(A1))
        # test_loss = sess.run(loss, feed_dict={x_data: x_vals_test, y_target: y_vals_test})
        # test_loss_vec.append(test_loss)
        # print('Step #' + str(i + 1) + ': Test Loss = ' + str(test_loss))

# Get and save the model parameter
file = h5py.File('modelW.h5', 'w')
file.create_dataset('weight', data=sess.run(A1))
file.close()

plt.plot(train_loss_vec, 'k-', label='Train Loss')
plt.plot(test_loss_vec, 'r-', label='Test Loss')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()
print(sess.run(A1))
