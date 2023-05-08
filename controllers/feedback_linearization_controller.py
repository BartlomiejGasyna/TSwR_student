import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        M = self.model.M(x)
        C = self.model.C(x)

        q1, q2, q1_dot, q2_dot = x

        e = q_r - [q1, q2]
        e_dot = q_r_dot - [q1_dot, q2_dot]


        Kd = np.array([[25, 0], [0, 30]])
        Kp = np.array([[20, 0], [0, 20]])
        v = q_r_ddot + Kd @ e_dot + Kp @ e
        tau = M@v + C@q_r_dot
        return tau