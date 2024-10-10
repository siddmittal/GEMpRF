import numpy as np
import matplotlib.pyplot as plt

# Define the range of x values
x_range = np.linspace(0, 8, 100)

# Define the value of c
c = 2

# Calculate f(x) = x + 2xc
f_x = x_range + x_range * c

# Calculate the derivative of f(x) with respect to x
derivative_x = 1 + c  # The derivative of x with respect to x is 1, and the derivative of 2xc with respect to x is 2c

# Calculate the derivative of f(x) with respect to c
derivative_c = x_range  # The derivative of 2xc with respect to c is 2x

# Create a plot to visualize f(x) and its derivatives
plt.figure(figsize=(8, 6))

plt.plot(x_range, f_x, label='f(x) = x + 2xc', color='blue')
plt.plot(x_range, [derivative_x] * len(x_range), label='Derivative w.r.t. X', color='orange')
plt.plot(x_range, derivative_c, label='Derivative w.r.t. c', color='red')
plt.xlabel('X')
plt.ylabel('Values')
plt.title('f(x) = x + 2xc and Its Derivatives')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

print('done')
