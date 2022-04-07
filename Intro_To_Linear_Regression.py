import numpy as np
import HyperplaneProcedures as hp

# Problem 2.1
X = np.array([[1, 1, 3, 3],
              [3, 1, 2, 6]])

th_blue = np.array([[-1, 1]]).T
th0_blue = 0

dim, num_samples = X.shape
blue_sq_errors = np.zeros(num_samples)
for i in range(num_samples):
    blue_sq_errors[i] = (X[0, i] - X[1, i]) ** 2
print("Problem 2.1: The squared errors for the blue line are: ", blue_sq_errors.tolist())




