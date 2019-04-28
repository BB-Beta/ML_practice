import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def main():
    mnist_data = input_data.read_data_sets('../data/MNIST_DATA', one_hot = True)
    x = tf.placeholder(dtype=tf.float32, shape=[None, 28*28], name = 'x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name = 'y')
    WD = tf.Variable(tf.zeros(shape=[28*28, 10], dtype=tf.float32), name='W')
    bD = tf.Variable(tf.zeros(shape=[10], dtype=tf.float32), name='b')
    logits = tf.add(tf.matmul(x, WD), bD, name='logits')
    label = tf.argmax(logits, 1, name='label')
    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits=logits))
    opt = tf.train.GradientDescentOptimizer(0.5)
    train_op = opt.minimize(loss_value)
    
    correct_value = tf.equal(tf.argmax(y, 1), label)
    correct_rate = tf.reduce_mean(tf.cast(correct_value, tf.float32))
    init_op = tf.global_variables_initializer()
    sv = tf.train.Supervisor(logdir = './log', init_op = init_op)
    saver = sv.saver
    with sv.managed_session(master='') as session:
        for index in range(2000):
            batch_x, batch_y = mnist_data.train.next_batch(100)
            _, batch_loss = session.run([train_op, loss_value], feed_dict={x: batch_x, y: batch_y})
            print('index: %d, loss: %f'%(index, batch_loss))
            if index % 100 == 0:
                saver.save(session, save_path='./log/best')
                print(session.run(bD, feed_dict={x:mnist_data.test.images}))
            accuracy = session.run(correct_rate, feed_dict={x:mnist_data.test.images, y:mnist_data.test.labels})
            print(session.run(bD, feed_dict={x:mnist_data.test.images}))
            print('accutacy:%f'%(accuracy))
if __name__ == '__main__':
    main()
    