import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the JSON file
with open(r"D:\workspace\ND-NeRF\exp\ABC\00000843_sigma0.1\emap\results\parametric_edges.json", "r") as file:
    data = json.load(file)

# Extract the control points for the curves and the endpoints for the lines
curves_ctl_pts = data.get("curves_ctl_pts", [])
lines_end_pts = data.get("lines_end_pts", [])

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the parametric curves
for curve in curves_ctl_pts:
    if len(curve) == 4:  # Ensure there are exactly 4 control points
        curve = np.array(curve)
        # Use interpolation to create smooth curves
        t = np.linspace(0, 1, 100)
        interpolated_curve = (
            (1 - t)[:, None] ** 3 * curve[0] +
            3 * (1 - t)[:, None] ** 2 * t[:, None] * curve[1] +
            3 * (1 - t)[:, None] * t[:, None] ** 2 * curve[2] +
            t[:, None] ** 3 * curve[3]
        )
        ax.plot(interpolated_curve[:, 0], interpolated_curve[:, 1], interpolated_curve[:, 2], label="Curve")

# Plot the lines
for line in lines_end_pts:
    if len(line) == 6:  # Ensure the line has exactly six values (two points in 3D)
        x, y, z = [line[i:i + 3] for i in range(0, 6, 3)]
        ax.plot([x[0], x[1]], [y[0], y[1]], [z[0], z[1]], 'r-', label="Line")

# Set plot labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Edge Visualization")

# Show legend and plot
ax.legend()
plt.show()
