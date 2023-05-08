import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        self.L = np.array([[3*p[0], 0],
                           [0, 3*p[1]],
                           [3*p[0]**2, 0],
                           [0, 3*p[1]**2],
                           [p[0]**3, 0],
                           [0, p[1]**3]])
        
        self.W = np.zeros((2, 6))
        self.W[0:2, 0:2] = np.eye(2)

        self.A = np.zeros((6,6))
        
        self.A[2:4, 4:6] = np.eye(2)
        self.A[0:2, 2:4] = np.eye(2)

        self.B = np.zeros((6, 2))

        self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    # def __init__(self, Tp, q0, Kp, Kd, p):
    #     self.model = ManipulatorModel(Tp)
    #     self.Kp = Kp
    #     self.Kd = Kd
    #     p1 = p[0]
    #     p2 = p[1]
    #     self.L = np.array([[3*p1, 0],
    #                        [0, 3*p2],
    #                        [3*p1**2, 0],
    #                        [0, 3*p2**2],
    #                        [p1**3, 0],
    #                        [0, p2**3]])
    #     self.W = np.zeros((2, 6))
    #     self.W[0, 0] = 1
    #     self.W[1, 1] = 1
    #     self.A = np.zeros((6, 6))
    #     self.A[0, 2] = 1
    #     self.A[1, 3] = 1
    #     self.A[2, 4] = 1
    #     self.A[3, 5] = 1
    #     self.B = np.zeros((6, 2))
    #     self.eso = ESO(self.A, self.B, self.W, self.L, q0, Tp)
    #     self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)
        C = self.model.C(x)
        M_in = -(M_inv @ C)
        self.A[2:4, 2:4] = M_in

        self.eso.A = self.A

        self.B[2:4, 0:2] = M_inv
        self.eso.B = self.B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        M_matrix = self.model.M(x)
        C_matrix = self.model.C(x)

        z_estimate = self.eso.get_state()
        x_estimate = z_estimate[0:2]
        x_estimate_dot = z_estimate[2:4]
        f = z_estimate[4:]
        print("z_estimate: ", z_estimate)
        print("x_estimate: ", x_estimate)
        print("x_estimate_dot: ", x_estimate_dot)
        print("f: ", f)

        e = q - q_d
        e_dot = x_estimate_dot - q_d_dot

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e
        u = M_matrix @ (v - f) + C_matrix @ x_estimate_dot

        self.update_params(x_estimate, x_estimate_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1))
        return u

