import numpy as np
from unittest import TestCase

iX = 0
iV = 1
NUMVAR = iV + 1

class KF:
    def __init__(self,  initial_X: float,
                        initial_v : float,
                        accel_variance: float) -> None:

        self._X = np.zeros(NUMVAR)
        self._X[iX] = initial_X
        self._X[iV] = initial_v
        # self._X = np.array([initial_X,initial_v])
        self._accel_variance = accel_variance


        self._P = np.eye(NUMVAR)

    def predict(self,dt:float) -> None:
        # _X = F * _X
        # P = F P Ft + G Gt a
        F = np.eye(NUMVAR)
        F[iX,iV] = dt
        # F = np.array([[1,dt],[0,1]])
        new_x = F.dot(self._X)

        G = np.zeros((2,1))
        G[iX] = 0.5 * dt**2
        G[iV] = dt
        # G = np.array([0.5*dt**2, dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T)*self._accel_variance

        self._P = new_P
        self._X = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - Hx
        # S = H P Ht + R
        # K = P Ht S ^-1
        # X = x + K y
        # P = (I- K H) * P

        H = np.array([1,0]).reshape((1,2))

        z = np.array([meas_value])
        R = np.array([meas_variance])
                                                                                                                                               
        y = z - H.dot(self._X)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._X + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._X = new_x

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._X

    @property
    def pos(self) -> float:
        return self._X[iX]

    @property
    def vel(self) -> float:
        return self._X[iV]
