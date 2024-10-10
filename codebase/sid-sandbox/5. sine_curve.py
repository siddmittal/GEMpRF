# sin(omega * t): https://www.oreilly.com/library/view/technical-mathematics-sixth/9780470534922/9780470534922_the_sine_wave_as_a_function_of_time.html
import numpy as np
import matplotlib.pyplot as plt
import math

# Take formula delta-t <= 1/(5*omega) from SO answer "https://stackoverflow.com/a/76895268/2386113"
frequency = 2 #Two HZ
# delta_t = 1/(2*frequency)
delta_t = max(math.floor( 1/(5*2*np.pi*frequency)), 1)

total_duration = 300
num_points = math.ceil(total_duration/delta_t)
t = np.linspace(0, total_duration, 600)
sine_t = np.sin(2*np.pi*frequency*t) #sin(omega * t)

#Only for test: to plot a line of constant amplitude
straight_line = t/t

# red for numpy.sin()
plt.plot(t, sine_t, color = 'red')
plt.plot(t, straight_line, color = 'blue')
plt.title("numpy.sin()")
plt.xlabel("time")
plt.ylabel("sin(t)")
plt.show()