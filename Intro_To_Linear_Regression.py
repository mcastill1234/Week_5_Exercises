import numpy as np
import HyperplaneProcedures as hp

# Problem 2.1
X = np.array([[1, 1, 3, 3],
              [3, 1, 2, 6]])


dim, num_samples = X.shape
blue_sq_errors = np.zeros(num_samples)
for i in range(num_samples):
    blue_sq_errors[i] = (X[0, i] - X[1, i]) ** 2
print("Problem 2.1: The squared errors for the blue line are: ", blue_sq_errors.tolist())


# Problem 2.2
th_blue = 1
th0_blue = 0


def grad_ms_error_th(x, y, th, th0):
    return -2 * (y - th * x - th0) * x


def grad_ms_error_th0(x, y, th, th0):
    return -2 * (y - th * x - th0)


blue_grad_contributions = []
for i in range(num_samples):
    blue_grad_contributions.append((grad_ms_error_th(X[0, i], X[1, i], th_blue, th0_blue),
                                    grad_ms_error_th0(X[0, i], X[1, i], th_blue, th0_blue)))
print("Problem 2.2: The gradient contributions from each point are: ", blue_grad_contributions)


# Problem 2.3
green_sq_errors = np.zeros(num_samples)
for i in range(num_samples):
    green_sq_errors[i] = (X[0, i] - X[1, i] + 1) ** 2
print("Problem 2.3: The squared errors for the green line are: ", green_sq_errors.tolist())


# Problem 2.4
th_red = 1
th0_red = 1
green_grad_contributions = []
for i in range(num_samples):
    green_grad_contributions.append((grad_ms_error_th(X[0, i], X[1, i], th_red, th0_red),
                                   grad_ms_error_th0(X[0, i], X[1, i], th_red, th0_red)))
print("Problem 2.4: The gradient contributions from each point are: ", green_grad_contributions)

# Problem 2.5
print("The mean squared error of the blue line is: ", sum(blue_sq_errors)/len(blue_sq_errors))
print("The mean squared error of the green line is: ", sum(green_sq_errors)/len(green_sq_errors))

