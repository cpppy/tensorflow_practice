import tensorflow as tf

if __name__=='__main__':

    mat1 = tf.constant([[3.0, 3.0]])
    mat2 = tf.constant([[2.0], [2.0]])
    product = tf.matmul(mat1, mat2)

    print(mat1)
    print(mat2)
    print(product)

    # execute ops
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close()

    # exec 2
    with tf.Session() as sess:
        result = sess.run(product)
        print(result)

    # exec 3
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            mat1 = tf.constant([[3.0, 3.0]])
            mat2 = tf.constant([[2.0], [2.0]])
            product = tf.matmul(mat1, mat2)
            result = sess.run(product)
            print(result)