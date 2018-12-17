import os
import tensorflow as tf
import tensorlayer as tl
from datautil import train_input_fn

if __name__ == '__main__':
    train_img_dir = '../img/train'
    train_img_arr = [train_img_dir + '/' + x for x in os.listdir(train_img_dir)]
    data_fn = train_input_fn(train_img_arr)

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        input_img = tf.placeholder(tf.float32, [None, 224, 224, 3])
        vgg = tl.models.VGG16(input_img, end_with='conv5_3')

    sess = tf.InteractiveSession()
    sess.run(tf.initializers.global_variables())
    vgg.restore_params(sess)

    temp_data = sess.run(data_fn)
    print(temp_data.shape)

    result = sess.run(vgg.outputs, feed_dict={input_img: temp_data})
    print(result.shape)
