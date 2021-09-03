import heterocl as hcl
import numpy as np
import time


class DubinsCar:
    def __init__(self, x=[0, 0, 0], wMax=1, speed=1, dMax=[0, 0, 0], uMode="min", dMode="max"):
        self.x = x
        self.wMax = wMax
        self.speed = speed
        self.dMax = dMax
        self.uMode = uMode
        self.dMode = dMode

    def opt_ctrl(self, t, state, spat_deriv):
        opt_w = hcl.scalar(self.wMax, "opt_w")
        # Just create and pass back, even though they're not used
        in3 = hcl.scalar(0, "in3")
        in4 = hcl.scalar(0, "in4")
        with hcl.if_(self.uMode == "min"):
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = -opt_w
            with hcl.else_():
                opt_w[0] = opt_w
        with hcl.elif_(self.uMode == "max"):
            with hcl.if_(spat_deriv[2] >= 0):
                opt_w[0] = opt_w
            with hcl.else_():
                opt_w[0] = -opt_w
        return opt_w[0], in3[0], in4[0]

    def optDstb(self, t, state, spat_deriv):
        """
        :param spat_deriv: tuple of spatial derivative in all dimensions
        :return: a tuple of optimal disturbances
        """
        # Graph takes in 4 possible inputs, by default, for now
        d1 = hcl.scalar(0, "d1")
        d2 = hcl.scalar(0, "d2")
        # Just create and pass back, even though they're not used
        d3 = hcl.scalar(0, "d3")
        return (d1[0], d2[0], d3[0])

    def dynamics(self, t, state, uOpt, dOpt):
        x_dot = hcl.scalar(0, "x_dot")
        y_dot = hcl.scalar(0, "y_dot")
        theta_dot = hcl.scalar(0, "theta_dot")

        x_dot[0] = self.speed * hcl.cos(state[2])
        y_dot[0] = self.speed * hcl.sin(state[2])
        theta_dot[0] = uOpt[0]

        return x_dot[0], y_dot[0], theta_dot[0]

    def dynamics_non_hcl(self, t, state, u, d):
        x_dot = self.speed * np.cos(state[2]) + d
        y_dot = self.speed * np.sin(state[2]) + d
        theta_dot = u + d

        return x_dot, y_dot, theta_dot

    def opt_ctrl_non_hcl(self, t, state, spat_deriv):
        opt_w = self.wMax
        if self.uMode == "min":
            if spat_deriv[2] >= 0:
                opt_w = -opt_w
            else:
                opt_w = opt_w
        if self.uMode == "max":
            if spat_deriv[2] >= 0:
                opt_w = opt_w
            else:
                opt_w = -opt_w
        return opt_w

    def optDstb_non_hcl(self, t, state, spat_deriv):
        return 0.0

    def update_state(self):
        """
        Make sure state is within grid bounds
        """
        if self.x[2] < -np.pi:
            self.x[2] += 2 * np.pi
        elif self.x[2] > np.pi:
            self.x[2] -= 2 * np.pi
