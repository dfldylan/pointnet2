# import tensorflow as tf
# import numpy as np
# from tf_grouping import query_ball_point, group_point, knn_point
# import time
#
# factor = 3
#
# class GroupPointTest(tf.test.TestCase):
#   def test(self):
#     pass
#
#   def test_grad(self):
#     with tf.device('/gpu:0'):
#       points = tf.constant(np.random.random((1,128*factor,16)).astype('float32'))
#       print points
#       xyz1 = tf.constant(np.random.random((1,128*factor,3)).astype('float32'))
#       xyz2 = tf.constant(np.random.random((1,8*factor,3)).astype('float32'))
#       radius = 0.3
#       nsample = 32
#       idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
#       # _,idx = knn_point(nsample, xyz1, xyz2)
#       grouped_points = group_point(points, idx)
#       print grouped_points
#
#     with self.test_session():
#       print "---- Going to compute gradient error"
#       err = tf.test.compute_gradient_error(points, (1,128*factor,16), grouped_points, (1,8*factor,32,16))
#       print err
#       self.assertLess(err, 1e-4)
#
# if __name__=='__main__':
#   tf.test.main()

import tensorflow as tf
import numpy as np
from tf_grouping import query_ball_point, group_point, knn_point
import time

if __name__ == '__main__':
    factor = 2097152
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            xyz1 = tf.placeholder(tf.float32, (1, 128 * factor, 3), 'xyz1')
            xyz2 = tf.placeholder(tf.float32, (1, 8 * factor, 3), 'xyz2')

            radius = 0.3
            nsample = 32
            idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
            # _,idx = knn_point(nsample, xyz1, xyz2)
        with tf.Session() as sess:
            for i in range(2):
                _xyz1 = np.random.random((1, 128 * factor, 3)).astype('float32')
                _xyz2 = np.random.random((1, 8 * factor, 3)).astype('float32')
                time0 = time.time()
                sess.run([idx], feed_dict={xyz1: _xyz1, xyz2: _xyz2})
                print time.time() - time0
