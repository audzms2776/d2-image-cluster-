import tensorflow as tf

BATCH_SIZE = 1


def read_tf_img(name):
    temp = tf.read_file(name)
    temp = tf.image.decode_image(temp, channels=3)
    temp = tf.reshape(temp, [300, 300, 3])
    temp = tf.image.resize_images(temp, [224, 224])
    temp = tf.cast(temp, tf.float32) * (1. / 255) - 0.5

    return temp


def _parse_function(img_name):
    lr_img = read_tf_img(img_name)

    return lr_img


def train_input_fn(img_arr):
    return tf.data.Dataset \
        .from_tensor_slices((img_arr,)) \
        .repeat() \
        .map(_parse_function, num_parallel_calls=4) \
        .batch(BATCH_SIZE) \
        .make_one_shot_iterator() \
        .get_next()
    # .prefetch(buffer_size=BATCH_SIZE) \

    # .batch(10, drop_remainder=True)
