import numpy as np
import pandas as pd
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

num_epoch = 200
learning_rate = .1
n = 2


"""Data Set is downloaded from Kaggle"""
"""https://www.kaggle.com/quantbruce/real-estate-price-prediction#Real%20estate.csv"""


def import_dataset():
    data = np.array(pd.read_csv("./data/real_estate.csv", header=None).values)
    new_data = data[1:, 0:].astype(np.float32)
    print(data.shape)

    data_2 = new_data[0:, 2:3]
    data_3 = new_data[0:, 3:4]
    new_data[:, 2:3] = (float(data_2.max(axis=0)[0]) - data_2) / float(data_2.max(axis=0)[0])
    new_data[:, 3:4] = (float(data_3.max(axis=0)[0]) - data_3) / float(data_3.max(axis=0)[0])

    batch_x_train = new_data[0:302, 3:5].transpose()
    batch_x_test = new_data[302:, 3:5].transpose()
    batch_y_train = new_data[0:302, 7:].transpose()
    batch_y_test = new_data[302:, 7:].transpose()

    # show_data(batch_x, batch_y, name="Generated Data")
    return batch_x_train, batch_y_train, batch_x_test, batch_y_test


def show_data(x, y, y_pred_b=None, name=None):
    plt.plot(x, y, 'bo', label=name)
    if y_pred_b is not None:
        plt.plot(x, y_pred_b, color="red", label="Fitted Line")
    plt.xlim(0, 2)
    plt.ylim(0, 4)
    plt.legend()
    plt.show()


def linear_regression(m, d):
    x = tf.placeholder(tf.float32, shape=(n, None), name="x")
    y = tf.placeholder(tf.float32, shape=(1, None), name="y")
    x_scaled = (x - m) / d

    with tf.variable_scope("lin_reg"):
        # Model parameters
        w = tf.get_variable("W", shape=(1, n))
        b = tf.get_variable("b", shape=())

        # Model Output
        y_predc = tf.matmul(w, x) + b

        loss = tf.reduce_sum(tf.square(y_predc - y))
    return x, y, y_predc, loss


def run_linear_regression():
    x_batch, y_batch, x_batch_test, y_batch_test = import_dataset()
    means = x_batch.mean(axis=1)
    deviations = x_batch.std(axis=1)
    x, y, y_pred, loss = linear_regression(means, deviations)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        feed_dict = {x: x_batch, y: y_batch}

        for i in range(num_epoch):
            p = session.run(train_op, feed_dict)
            print(i, "loss:", loss.eval(feed_dict))

        print('Predicting')
        y_pred_batch = session.run(y_pred, {x: x_batch_test})
        loss_test = np.sqrt(np.square(y_pred_batch - y_batch_test)/y_batch_test.shape[1])

        print(i, "Test loss:", np.mean(loss_test))


if __name__ == "__main__":
    run_linear_regression()
    # import_dataset()
