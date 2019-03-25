import tensorflow as tf

if __name__=='__main__':

    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
        output_res = sess.run([output], feed_dict={input1:[8.0], input2:[2.0]})
        print(output_res)










