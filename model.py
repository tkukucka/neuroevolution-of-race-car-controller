import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable

from curved_intersection import track_curved_intersection
from straight_intersection import track_straight_intersection

colormap = plt.get_cmap("jet")

# Function to compute new velocity after a time interval dt
def velocity_function(v_1, gas, dt):  # both velocity and gas in range from 0 to 1, dt must not be larger than 1
    mass_constant = 3
    v_min, v_max = 0.1, 1
    return v_1 + (gas - (v_1-v_min)/(v_max-v_min))/mass_constant*dt

# used to assign color for different kinds of plot
def color_function(color, style, inside_plt):
    if style == None:
        # plotting inside track blue and outside red
        color = "r"
        if inside_plt == 1:
            color = "b"
        # color define manually
    if style in ["r", "g", "b"]:
        color = style
        style = "-"
    return color, style

def next_pos(x_1, y_1, attitude_1, steer_angle, velocity, dt, track, sensor_settings=[[-1, -0.5, 0, 0.5, 1], [15]*5],
             plot=False, plot_sensor=False, distance_steps=None):

    sensor_angles = sensor_settings[0]
    view_distances = sensor_settings[1]

    if distance_steps is not None:  # use a constant distance stepping scheme
        distance = distance_steps
        if velocity == 0:  # this is only used for getting sensor input for the first time step, otherwise v is never 0
            time_step = distance/1.5
            distance = 0
        else:
            time_step = distance/velocity
    else:
        distance = velocity*dt
        time_step = dt

    arc = abs(steer_angle) > 1e-14  # avoid numerical diffusion when steering angle is very low

    # if trajectory is an arc
    if arc:
        r = 1/steer_angle  # compute turn radius
        x_c = x_1 + r*np.cos(attitude_1)  # x_c and y_c are the coordinates of the center of circle forming the arc
        y_c = y_1 - r*np.sin(attitude_1)
        turn_angle = distance*steer_angle  # angle of the arc traced out
        attitude_2 = attitude_1 + turn_angle  # direction car faces at end of turn
        x_2 = x_c - r*np.cos(attitude_2)  # x_2 and y_2 is the new position
        y_2 = y_c + r*np.sin(attitude_2)

        # find intersection points between trajectory and track
        intersections = track_curved_intersection(x_c, y_c, r, attitude_1, attitude_2, track)

    # if trajectory is a straight line
    else:  # different procedure for straight segments to avoid division by zero
        attitude_2 = attitude_1  # new attitude is the same
        x_2 = x_1 + distance*np.sin(attitude_1)  # compute new position of car
        y_2 = y_1 + distance*np.cos(attitude_1)

        # find intersection points between trajectory and track
        intersections = track_straight_intersection((x_1, y_1), (x_2, y_2), track)

    # compute sensor outputs
    sensor_output = []
    for i in range(len(sensor_angles)):
        sensor_angle = sensor_angles[i]
        view_distance = view_distances[i]
        view_angle = -attitude_2 + sensor_angle + np.pi/2
        x_view = x_2 + view_distance*np.cos(view_angle)
        y_view = y_2 + view_distance*np.sin(view_angle)

        # compute intersections between sensor line and track segments
        sensor_intersections = track_straight_intersection((x_2, y_2), (x_view, y_view), track)

        # select closest intersection from sensor line
        if len(sensor_intersections) > 1:
            intersection_distances = []
            for i in sensor_intersections:
                intersection_distances.append(i[1])
            closest = np.argmin(intersection_distances)
            sensor_intersections = sensor_intersections[closest]
        elif len(sensor_intersections) == 1:
            sensor_intersections = sensor_intersections[0]

        if len(sensor_intersections) > 0:
            sensor_output.append(1-sensor_intersections[1])
            if plot_sensor and plot:
                plt.plot([x_2, sensor_intersections[0][0]], [y_2, sensor_intersections[0][1]], c="gray")
        else:
            sensor_output.append(0)
            if plot_sensor and plot:
                plt.plot([x_2, x_view], [y_2, y_view], c="gray")

    # determine if wall was hit
    hit_wall = True if len(intersections) > 0 else False

    # compute centripetal acceleration
    a_c = velocity**2/abs(r) if arc else 0

    # plotting trajectory
    if plot:
        v_max, v_min = 15, 1.5
        frac = (velocity - v_max) / (v_min - v_max)
        color = colormap(1 - frac)
        plt.scatter([x_1, x_2], [y_1, y_2], c="k")
        if arc:
            x_plt_0, y_plt_0 = x_1, y_1
            attitude_plt = attitude_1
            n_plot_pts = 50
            for i in range(n_plot_pts):
                attitude_plt += turn_angle/n_plot_pts
                x_plt_1 = x_c - r*np.cos(attitude_plt)
                y_plt_1 = y_c + r*np.sin(attitude_plt)
                plt.plot([x_plt_0, x_plt_1], [y_plt_0, y_plt_1], c=color)
                x_plt_0, y_plt_0 = x_plt_1, y_plt_1
        else:
            plt.plot([x_1, x_2], [y_1, y_2], c=color)

    return x_2, y_2, attitude_2, hit_wall, a_c, sensor_output, time_step


# wrapper function for next_pos, (core of the model): takes state and inputs, gives next state after one time step
def step(state_variables, input_variables, track, dt=1.0, max_velocity=15, max_steer_angle=0.25, a_max=13, plot=False,
         distance_step=None, plot_sensor=False, sensor_settings=None):

    steering, pedal = input_variables
    x, y, theta, attitude, velocity, angle_covered, acceleration_penalty = state_variables

    velocity = velocity_function(velocity, pedal, dt)

    # update position
    x, y, attitude, hit_wall, a_c, sensor_output, time_step = \
        next_pos(x, y, attitude, steering*max_steer_angle, velocity*max_velocity, dt, track, plot=plot,
                 plot_sensor=plot_sensor, distance_steps=distance_step, sensor_settings=sensor_settings)

    # update angle covered
    theta_new = np.arctan2(y, -x)
    if theta_new < 0:  # make theta be between 0 - 2*pi
        theta_new += 2 * np.pi
    if theta_new < 0.5 * np.pi and theta > 1.5 * np.pi:  # check if theta passes through 0
        angle_covered += theta_new - theta + 2 * np.pi
    elif theta < 0.5 * np.pi and theta_new > 1.5 * np.pi:  # check if theta passes through 0 backwards
        angle_covered += theta_new - theta - 2 * np.pi
    else:
        angle_covered += theta_new - theta
    theta = theta_new

    # compute acceleration penalty
    if a_c > a_max:
        acceleration_penalty[0] += (a_c - a_max)*dt
    acceleration_penalty[1] += a_c*dt

    end = True if hit_wall else False

    return x, y, theta, attitude, velocity, angle_covered, acceleration_penalty, end, sensor_output, time_step


def evaluate(net, track, laps=False, a_max=13, trajectory_length=500, start_pos=(-60, 0), dt=1.0, pedal=False,
             plot=False, plot_sensor=False, distance_step=None, sensor_settings=None):

    if plot:
        plt.figure(figsize=(5.5, 4.5), dpi=120)

    # initialize variables
    y_max = max(track.y_in)
    attitude = 0
    angle_covered = 0
    drive_time = 0
    acceleration_penalty = [0, 0]  # integral of exceeded centripetal a, integral of total centripetal a
    v_max_reached = 0
    x, y = start_pos
    theta = np.arctan2(y, -x)  # this variable is passed on to save time as it needs to be computed every iteration
    if theta < 0:  # make theta be between 0 - 2*pi
        theta += 2 * np.pi

    # evaluate initial sensor output by making step of zero size
    state_variables = x, y, theta, attitude, 0, angle_covered, acceleration_penalty
    out = step(state_variables, [0, 0], track, plot=plot, dt=dt, plot_sensor=plot_sensor, distance_step=distance_step,
               sensor_settings=sensor_settings)
    sensor_output = out[8]

    # set velocity to max if pedal function is disabled
    velocity = 0.1 if pedal else 0.5
    state_variables = x, y, theta, attitude, velocity, angle_covered, acceleration_penalty

    # Run simulation in a loop, activate network in each step to get input
    for i in range(trajectory_length):
        # Single net design
        if net[0] is None:
            input_state = sensor_output + [velocity]
            net_output = net[1].activate(input_state)
            steering = net_output[0] - 0.5
            pedal_setting = net_output[1] if pedal else 0.5

        # Double net design
        else:
            # activate steering network
            input_state = sensor_output
            steering = net[0].activate(input_state)[0] - 0.5
            # activate pedal network
            if pedal:
                input_state = steering, velocity, sensor_output[1]
                pedal_setting = net[1].activate(input_state)[0]
            else:
                pedal_setting = 0.5

        out = step(state_variables, [steering, pedal_setting], track, plot=plot, dt=dt, plot_sensor=plot_sensor,
                   a_max=a_max, distance_step=distance_step, sensor_settings=sensor_settings)
        state_variables = out[:7]
        sensor_output = out[8]
        time_step = out[9]

        # compute maximum velocity that was reached
        if out[4] > v_max_reached:
            v_max_reached = out[4]

        drive_time += time_step
        # stopping criteria for simulation
        if out[7]:  # if wall is hit
            break
        if laps:  # if car travels more than one lap in lap mode
            if out[5] > np.pi*2:
                break
        else:  # if car travels entire track in up mode
            if out[1] >= y_max:
                break

    x, y, theta, attitude, velocity, angle_covered, acceleration_penalty, = state_variables

    # CHANGE REWARD FUNCTION HERE
    if laps:
        reward = angle_covered * 50 + angle_covered / drive_time * 1000
        if pedal:
            reward -= acceleration_penalty[0]*10 + acceleration_penalty[1]/drive_time*10 - v_max_reached*25
        else:
            reward -= acceleration_penalty[1]/drive_time
    else:
        reward = y/2 - drive_time*2 - acceleration_penalty[0]*2

    if plot:
        track.plot()
        plt.axis("equal")
        plt.colorbar(ScalarMappable(cmap=colormap), label=r"$V/V_{max}$", fraction=0.05)
        plt.xlabel(r"$x$ [m]")
        plt.ylabel(r"$y$ [m]")
        plt.savefig("last_out.png")
        plt.show()
        print("Reward of plotted trajectory =", reward, "Acceleration penatly =", acceleration_penalty)
        print("Time driven =", drive_time, "Distance covered =", angle_covered, "V_max =", v_max_reached*15)

    return reward

if __name__ == "__main__":
    pass