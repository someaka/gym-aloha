import numpy as np


def sample_box_pose(seed=None):
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose(seed=None):
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


def sample_screwdriver_pose(seed=None):
    # Screwdriver - positioned on the right side 
    x_range = [0.18, 0.28]
    y_range = [0.53, 0.63]
    z_range = [0.2, 0.22]  # Higher above the plate for larger objects

    rng = np.random.RandomState(seed)

    ranges = np.vstack([x_range, y_range, z_range])
    screwdriver_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    screwdriver_quat = np.array([1, 0, 0, 0])
    screwdriver_pose = np.concatenate([screwdriver_position, screwdriver_quat])

    # Bolt - positioned on the left side
    x_range = [-0.12, -0.02]
    y_range = [0.53, 0.63]
    z_range = [0.2, 0.22]  # Higher above the plate for larger objects

    ranges = np.vstack([x_range, y_range, z_range])
    bolt_position = rng.uniform(ranges[:, 0], ranges[:, 1])

    bolt_quat = np.array([1, 0, 0, 0])
    bolt_pose = np.concatenate([bolt_position, bolt_quat])

    # Nut - positioned above the bolt
    nut_position = bolt_position.copy()
    nut_position[2] = 0.25  # Higher z position for larger objects

    nut_quat = np.array([1, 0, 0, 0])
    nut_pose = np.concatenate([nut_position, nut_quat])

    # Plate - fixed position (this is not used since we removed the free joint from the plate)
    # But we keep it for consistency in the API
    plate_position = np.array([0.08, 0.58, 0.025])
    plate_quat = np.array([1, 0, 0, 0])
    plate_pose = np.concatenate([plate_position, plate_quat])

    return screwdriver_pose, bolt_pose, nut_pose, plate_pose
