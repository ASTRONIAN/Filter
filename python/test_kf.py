import numpy as np
# from KalmanF import KF
import KalmanF as python_kf
# from unittest import TestCase
import unittest

# == Import CPP wrapper .so file == #
import os
cur_file_path = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(cur_file_path + '/../cpp/build')

import kf_cpp as cpp_kf

# == End import CPP wrapper .so file == #

from parameterized import parameterized_class

@parameterized_class([
   { "KF": python_kf.KF },
   {"KF": cpp_kf.KF}
])

class TestKF(unittest.TestCase):
    def test_can_construct_with_x_and_v(self):
        x = 0.25
        v = 2.3

        kf = self.KF(initial_X=x,initial_v=v,accel_variance=1.2)
        self.assertAlmostEqual(kf.pos,x)
        self.assertAlmostEqual(kf.vel,v)

    def test_after_calling_predict_x_and_P_are_of_right_shape(self):
        x = 0.25
        v = 2.3

        kf = self.KF(initial_X=x,initial_v=v,accel_variance=1.2)
        kf.predict(dt=0.1)

        self.assertEqual(kf.cov.shape,(2,2))
        self.assertEqual(kf.mean.shape,(2,))

    def test_calling_predict_increase_state_uncertainty(self):
        x = 0.25
        v = 2.3

        kf = self.KF(initial_X=x,initial_v=v,accel_variance=1.2)

        for i in range(10):
            det_before = np.linalg.det(kf.cov)
            kf.predict(dt=0.1)
            det_after = np.linalg.det(kf.cov)

            self.assertGreater(det_after,det_before)
            print(det_before, det_after)

    def test_calling_update_does_not_crash(self):
        x = 0.25
        v = 2.3

        kf = self.KF(initial_X=x,initial_v=v,accel_variance=1.2)
        kf.update(meas_value=0.1,meas_variance=0.1)

    def test_calling_predict_increase_state_uncertainty(self):
        x = 0.25
        v = 2.3

        kf = self.KF(initial_X=x,initial_v=v,accel_variance=1.2)

        det_before = np.linalg.det(kf.cov)
        kf.update(meas_value=0.1,meas_variance=0.1)
        det_after = np.linalg.det(kf.cov)

        self.assertLess(det_after,det_before)
