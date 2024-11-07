import numpy as np
import pandas as pd

def compute_features(group):
    features = {}
    # Number of events
    
    # Extract move events
    move_events = group[group["event_type"] == "move"]
    features["num_moves"] = len(move_events)
    # Extract click events
    click_events = group[group["event_type"] == "click"]
    features["num_clicks"] = len(click_events)
    # Extract scroll events
    scroll_events = group[group["event_type"] == "scroll"]
    features["num_scrolls"] = len(scroll_events)

    # Click positions
    if len(click_events) >= 1:
        features["click_x_mean"] = click_events["x"].mean()
        features["click_y_mean"] = click_events["y"].mean()
    else:
        features["click_x_mean"] = 0
        features["click_y_mean"] = 0

    # Compute movement distance and related features
    if len(move_events) >= 2:
        move_events = move_events.sort_values("time")
        x = move_events["x"].values
        y = move_events["y"].values
        t = move_events["time"].values
        dx = np.diff(x)
        dy = np.diff(y)
        dt = np.diff(t)

        # Handle zeros in dt to prevent division by zero
        valid = dt != 0
        dx = dx[valid]
        dy = dy[valid]
        dt = dt[valid]
        distances = np.sqrt(dx**2 + dy**2)
        total_distance = np.sum(distances)
        features["movement_distance"] = total_distance

        if len(distances) > 0:
            # Velocity calculations
            velocities = distances / dt
            features["velocity_mean"] = np.mean(velocities)
            features["velocity_max"] = np.max(velocities)
            features["velocity_min"] = np.min(velocities)
            features["velocity_sd"] = np.std(velocities)
            features["velocity_x_mean"] = np.mean(dx / dt)
            features["velocity_y_mean"] = np.mean(dy / dt)
        else:
            # No valid velocities
            features.update({
                "velocity_mean": 0,
                "velocity_max": 0,
                "velocity_min": 0,
                "velocity_sd": 0,
                "velocity_x_mean": 0,
                "velocity_y_mean": 0,
            })
            velocities = np.array([])

        # Acceleration calculations
        if len(velocities) >= 2:
            dv = np.diff(velocities)
            dt_acc = dt[1:]  # Time intervals for acceleration
            valid_acc = dt_acc != 0
            dv = dv[valid_acc]
            dt_acc = dt_acc[valid_acc]
            if len(dv) > 0:
                dv_dt = dv / dt_acc
                features["acceleration_mean"] = np.mean(dv_dt)
                features["acceleration_max"] = np.max(dv_dt)
                features["acceleration_min"] = np.min(dv_dt)
                features["acceleration_sd"] = np.std(dv_dt)
            else:
                features.update({
                    "acceleration_mean": 0,
                    "acceleration_max": 0,
                    "acceleration_min": 0,
                    "acceleration_sd": 0,
                })
                dv_dt = np.array([])
        else:
            features.update({
                "acceleration_mean": 0,
                "acceleration_max": 0,
                "acceleration_min": 0,
                "acceleration_sd": 0,
            })
            dv_dt = np.array([])

        # Jerk calculations
        if len(dv_dt) >= 2:
            da = np.diff(dv_dt)
            dt_jerk = dt_acc[1:]
            valid_jerk = dt_jerk != 0
            da = da[valid_jerk]
            dt_jerk = dt_jerk[valid_jerk]
            if len(da) > 0:
                da_dt = da / dt_jerk
                features["jerk_mean"] = np.mean(da_dt)
                features["jerk_sd"] = np.std(da_dt)
            else:
                features.update({
                    "jerk_mean": 0,
                    "jerk_sd": 0,
                })
        else:
            features.update({
                "jerk_mean": 0,
                "jerk_sd": 0,
            })

        # Angular velocity calculations
        angles = np.arctan2(dy, dx)
        d_angle = np.diff(angles)
        d_angle = (d_angle + np.pi) % (2 * np.pi) - np.pi  # Normalize angles
        dt_ang = dt[1:]
        valid_ang = dt_ang != 0
        d_angle = d_angle[valid_ang]
        dt_ang = dt_ang[valid_ang]
        if len(d_angle) > 0:
            angular_velocity = d_angle / dt_ang
            features["angular_velocity_mean"] = np.mean(angular_velocity)
            features["angular_velocity_sd"] = np.std(angular_velocity)
        else:
            features.update({
                "angular_velocity_mean": 0,
                "angular_velocity_sd": 0,
            })
    else:
        features.update({
            "movement_distance": 0,
            "velocity_mean": 0,
            "velocity_max": 0,
            "velocity_min": 0,
            "velocity_sd": 0,
            "velocity_x_mean": 0,
            "velocity_y_mean": 0,
            "acceleration_mean": 0,
            "acceleration_max": 0,
            "acceleration_min": 0,
            "acceleration_sd": 0,
            "jerk_mean": 0,
            "jerk_sd": 0,
            "angular_velocity_mean": 0,
            "angular_velocity_sd": 0,
        })

    # Movement duration
    if len(move_events) >= 1:
        features["movement_duration"] = move_events["time"].max() - move_events["time"].min()
    else:
        features["movement_duration"] = 0

    # Pause time (idle cursor time)
    total_time = group["time"].max() - group["time"].min()
    features["total_time"] = total_time
    features["pause_time"] = total_time - features["movement_duration"]

    # Flips (directional changes)
    if len(move_events) >= 2 and len(dx) >= 2:
        features["flips_x"] = np.sum(np.diff(np.sign(dx)) != 0)
        features["flips_y"] = np.sum(np.diff(np.sign(dy)) != 0)
    else:
        features["flips_x"] = 0
        features["flips_y"] = 0

    # Number of pauses (idle periods)
    if len(move_events) >= 2 and len(dt) >= 1:
        idle_threshold = 0.2  # Define a threshold for idle time
        pauses = dt[dt > idle_threshold]
        features["pause_count"] = len(pauses)
    else:
        features["pause_count"] = 0

    # Hold time for clicks
    if len(click_events) >= 1:
        pressed_events = click_events[click_events["pressed"] == True]
        released_events = click_events[click_events["pressed"] == False]
        if len(pressed_events) == len(released_events):
            hold_times = released_events["time"].values - pressed_events["time"].values
            features["hold_time_mean"] = np.mean(hold_times) if len(hold_times) > 0 else 0
            features["hold_time_sd"] = np.std(hold_times) if len(hold_times) > 1 else 0
        else:
            features["hold_time_mean"] = 0
            features["hold_time_sd"] = 0
    else:
        features["hold_time_mean"] = 0
        features["hold_time_sd"] = 0

    return pd.Series(features)