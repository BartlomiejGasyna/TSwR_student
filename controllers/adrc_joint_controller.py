import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [self.b],
                      [0]])
        L = np.array([[3*p],
                      [3*p**2],
                      [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        B = np.array([[0],
                [self.b],
                [0]])
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot, b_vec_iter):
        ### TODO implement ADRC
        q = x[0]
        z_estimate = self.eso.get_state()
        x_estimate = z_estimate[0]
        x_estimate_dot = z_estimate[1]
        f = z_estimate[2]
        e = q - q_d
        e_dot = x_estimate_dot - q_d_dot
        v = q_d_ddot + self.kd * e_dot + self.kp * e
        u = (v - f) / self.b
        self.eso.update(q, u)
        print("z_estimate: ", z_estimate)
        
        self.l1 = 0.5
        self.r1 = 0.01
        self.m1 = 1.
        self.l2 = 0.5
        self.r2 = 0.01
        self.m2 = 1.
        self.I_1 = 1 / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1 / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = 0.0
        self.r3 = 0.01
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = self.I_1 + self.I_2 + self.m1 * d1 ** 2 + self.m2 * (self.l1 ** 2 + d2 ** 2) + \
        self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        delta = self.I_2 + self.m2 * d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        m_00 = alpha + 2 * beta * np.cos(x_estimate)
        m_01 = delta + beta * np.cos(x_estimate)
        m_10 = m_01
        m_11 = delta

        M_matrix = np.zeros((2, 2))

        M_matrix[0,0] = m_00
        M_matrix[0,1] = m_01
        M_matrix[1,0] = m_10
        M_matrix[1,1] = m_11

        M_matrix_inverse = np.linalg.inv(M_matrix)
        self.set_b(M_matrix_inverse[b_vec_iter, b_vec_iter])
        return u