import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

num_epoch = 80
learning_rate = .01


def generate_dataset():
    batch_x = np.linspace(0, 2, 100)
    batch_y = 1.5 * batch_x + np.random.randn(*batch_x.shape) * 0.3 + 0.6

    show_data(batch_x, batch_y)
    return batch_x, batch_y


def show_data(x, y):
    plt.plot(x, y, 'bo', "Generated Data")
    plt.legend()
    plt.show()


def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None, ), name="x")
    y = tf.placeholder(tf.float32, shape=(None, ), name="y")

    with tf.variable_scope("lin_reg") as scope:
        w = tf.Variable(np.random.normal(), name="w")
        b = tf.Variable(np.random.normal(), name="b")

        y_prdc = tf.add(tf.multiply(w, x), b)

        loss = tf.reduce_mean(tf.square(y_prdc - y))
    return x, y, y_prdc, loss


def run_linear_regression():
    x_batch, y_batch = generate_dataset()
    x, y, y_pred, loss = linear_regression()

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        feed_dict = {x: x_batch, y: y_batch}

        for i in range(num_epoch):
            p = session.run(train_op, feed_dict)
            print(i, "loss:", loss.eval(feed_dict))

        print('Predicting')
        y_pred_batch = session.run(y_pred, {x: x_batch})

    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch, color='red')
    plt.xlim(0, 2)
    plt.ylim(0, 4)
    plt.show()
    # plt.savefig('plot.png')


if __name__ == "__main__":
    run_linear_regression()
