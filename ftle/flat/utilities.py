


import matplotlib.pyplot as plt
import numpy as np


def RK4_advection_2d(velocity_points, velocity_vectors, trajectories, dt, fine_time, time_independent):

    if time_independent:
        interp_u = LinearNDInterpolator(velocity_points, velocity_vectors[:, 0], fill_value=0)
        interp_v = LinearNDInterpolator(velocity_points, velocity_vectors[:, 1], fill_value=0)
    
        for t_index, _ in enumerate(fine_time):
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]
    
            k1_x, k1_y = interp_u(x_curr, y_curr), interp_v(x_curr, y_curr)
            k2_x, k2_y = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y), \
                         interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y)
            k3_x, k3_y = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y), \
                         interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y)
            k4_x, k4_y = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y), \
                         interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y)
    
            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    
            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next

    else: 
        for t_index, t in enumerate(fine_time):
            t_floor = int(np.floor(t))
            t_ceiling = int(np.ceil(t))
            t_fraction = t - t_floor  
    
            # Interpolate velocity vectors at this time
            u_interp = interpolate(velocity_vectors[:, 0, t_floor], velocity_vectors[:, 0, t_ceiling], t_fraction)
            v_interp = interpolate(velocity_vectors[:, 1, t_floor], velocity_vectors[:, 1, t_ceiling], t_fraction)
    
            interp_u = LinearNDInterpolator(velocity_points, u_interp, fill_value=0)
            interp_v = LinearNDInterpolator(velocity_points, v_interp, fill_value=0)
    
            x_curr = trajectories[:, 0, t_index]
            y_curr = trajectories[:, 1, t_index]
    
            k1_x, k1_y = interp_u(x_curr, y_curr), interp_v(x_curr, y_curr)
            k2_x, k2_y = interp_u(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y), \
                         interp_v(x_curr + 0.5 * dt * k1_x, y_curr + 0.5 * dt * k1_y)
            k3_x, k3_y = interp_u(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y), \
                         interp_v(x_curr + 0.5 * dt * k2_x, y_curr + 0.5 * dt * k2_y)
            k4_x, k4_y = interp_u(x_curr + dt * k3_x, y_curr + dt * k3_y), \
                         interp_v(x_curr + dt * k3_x, y_curr + dt * k3_y)
    
            x_next = x_curr + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_next = y_curr + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    
            trajectories[:, 0, t_index + 1] = x_next
            trajectories[:, 1, t_index + 1] = y_next
        
    return trajectories




def interpolate(floor_data, ceiling_data, t_fraction):
    return t_fraction*ceiling_data + (1-t_fraction)*floor_data



def plot_FTLE_2d(particles, ftle):
    plt.figure(figsize=(6, 6))


    # Scatter plot of FTLE values with color mapping
    sc = plt.scatter(particles[:, 0], particles[:, 1], c=ftle, cmap='plasma', s=10)
    plt.colorbar(sc, label="FTLE Value")

    plt.title("FTLE Field at Initial Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.show()

    return None

def plot_FTLE_3d(particles, ftle, tol=0.01):
    """
    particles: (N, 3) array of particle positions in 3D
    ftle: (N,) array of FTLE values
    tol: float, tolerance for selecting slices near specific y values
    """
    y_values = particles[:, 1]
    y_slices = [np.min(y_values), np.median(y_values), np.max(y_values)]

    fig = plt.figure(figsize=(16, 10))

    # Full 3D plot (left side)
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    sc = ax3d.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                      c=ftle, cmap='plasma', s=10)
    ax3d.set_title("Full 3D FTLE Field")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    cbar = fig.colorbar(sc, ax=ax3d, shrink=0.6)
    cbar.set_label("FTLE Value")

    # Add 3 XZ slice subplots (right side)
    for i, y_val in enumerate(y_slices):
        ax = fig.add_subplot(2, 2, i + 2)  # Plots go into 2nd, 3rd, and 4th subplot spots
        mask = np.abs(particles[:, 1] - y_val) < tol
        if np.any(mask):
            slice_particles = particles[mask]
            slice_ftle = ftle[mask]
            sc2 = ax.scatter(slice_particles[:, 0], slice_particles[:, 2],
                             c=slice_ftle, cmap='plasma', s=10)
            ax.set_title(f"XZ Slice at y ≈ {y_val:.3f}")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
        else:
            ax.set_title(f"No points found near y = {y_val:.3f}")

    plt.tight_layout()
    plt.show()

    return None
