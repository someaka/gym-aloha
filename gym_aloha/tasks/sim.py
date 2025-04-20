import collections

import numpy as np
from dm_control.suite import base

from gym_aloha.constants import (
    START_ARM_POSE,
    normalize_puppet_gripper_position,
    normalize_puppet_gripper_velocity,
    unnormalize_puppet_gripper_position,
)

BOX_POSE = [None]  # to be changed from outside

"""
Environment for simulated robot bi-manual manipulation, with joint position control
Action space:      [left_arm_qpos (6),             # absolute joint position
                    left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                    right_arm_qpos (6),            # absolute joint position
                    right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

Observation space: {"qpos": Concat[ left_arm_qpos (6),         # absolute joint position
                                    left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                    right_arm_qpos (6),         # absolute joint position
                                    right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                    "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                    left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                    right_arm_qvel (6),         # absolute joint velocity (rad)
                                    right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                    "images": {"main": (480x640x3)}        # h, w, c, dtype='uint8'
"""


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[7 : 7 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[7 + 6]

        left_gripper_action = unnormalize_puppet_gripper_position(normalized_left_gripper_action)
        right_gripper_action = unnormalize_puppet_gripper_position(normalized_right_gripper_action)

        full_left_gripper_action = [left_gripper_action, -left_gripper_action]
        full_right_gripper_action = [right_gripper_action, -right_gripper_action]

        env_action = np.concatenate(
            [left_arm_action, full_left_gripper_action, right_arm_action, full_right_gripper_action]
        )
        super().before_step(env_action, physics)
        return

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        left_qpos_raw = qpos_raw[:8]
        right_qpos_raw = qpos_raw[8:16]
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [normalize_puppet_gripper_position(left_qpos_raw[6])]
        right_gripper_qpos = [normalize_puppet_gripper_position(right_qpos_raw[6])]
        return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        left_qvel_raw = qvel_raw[:8]
        right_qvel_raw = qvel_raw[8:16]
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [normalize_puppet_gripper_velocity(left_qvel_raw[6])]
        right_gripper_qvel = [normalize_puppet_gripper_velocity(right_qvel_raw[6])]
        return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos(physics)
        obs["qvel"] = self.get_qvel(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = {}
        obs["images"]["top"] = physics.render(height=480, width=640, camera_id="top")
        obs["images"]["angle"] = physics.render(height=480, width=640, camera_id="angle")
        obs["images"]["vis"] = physics.render(height=480, width=640, camera_id="front_close")

        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError


class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_right_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


class InsertionTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7 * 2 :] = BOX_POSE[0]  # two objects
            # print(f"{BOX_POSE=}")
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # return whether peg touches the pin
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        touch_right_gripper = ("red_peg", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper = (
            ("socket-1", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-2", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-3", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
            or ("socket-4", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        )

        peg_touch_table = ("red_peg", "table") in all_contact_pairs
        socket_touch_table = (
            ("socket-1", "table") in all_contact_pairs
            or ("socket-2", "table") in all_contact_pairs
            or ("socket-3", "table") in all_contact_pairs
            or ("socket-4", "table") in all_contact_pairs
        )
        peg_touch_socket = (
            ("red_peg", "socket-1") in all_contact_pairs
            or ("red_peg", "socket-2") in all_contact_pairs
            or ("red_peg", "socket-3") in all_contact_pairs
            or ("red_peg", "socket-4") in all_contact_pairs
        )
        pin_touched = ("red_peg", "pin") in all_contact_pairs

        reward = 0
        if touch_left_gripper and touch_right_gripper:  # touch both
            reward = 1
        if (
            touch_left_gripper and touch_right_gripper and (not peg_touch_table) and (not socket_touch_table)
        ):  # grasp both
            reward = 2
        if peg_touch_socket and (not peg_touch_table) and (not socket_touch_table):  # peg and socket touching
            reward = 3
        if pin_touched:  # successful insertion
            reward = 4
        return reward


class ScrewdriverTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set BOX_POSE from outside
        # reset qpos, control and object positions
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            # Set positions for screwdriver, bolt, and nut (plate is fixed)
            physics.named.data.qpos[-7 * 3 :] = BOX_POSE[0][:21]  # three objects with free joints
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # Return reward based on task progress
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # Check contacts between objects and grippers
        touch_right_gripper_screwdriver = ("screwdriver", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper_bolt = ("bolt", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_left_gripper_nut = ("nut", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        # Check contacts with table
        screwdriver_touch_table = ("screwdriver", "table") in all_contact_pairs
        bolt_touch_table = ("bolt", "table") in all_contact_pairs
        nut_touch_table = ("nut", "table") in all_contact_pairs

        # Check contact between objects
        screwdriver_touch_bolt = ("screwdriver", "bolt") in all_contact_pairs
        nut_touch_bolt = ("nut", "bolt") in all_contact_pairs

        # Check if bolt is in plate
        bolt_in_plate = ("bolt", "plate_hole") in all_contact_pairs

        # Define reward stages
        reward = 0
        if (touch_left_gripper_bolt or touch_left_gripper_nut) and touch_right_gripper_screwdriver:  # touch both
            reward = 1
        if (
            (touch_left_gripper_bolt or touch_left_gripper_nut) and touch_right_gripper_screwdriver and
            (not screwdriver_touch_table) and (not bolt_touch_table) and (not nut_touch_table)
        ):  # grasp both
            reward = 2
        if (screwdriver_touch_bolt or nut_touch_bolt) and (not screwdriver_touch_table) and (not bolt_touch_table) and (not nut_touch_table):  # objects touching
            reward = 3
        if bolt_in_plate:  # successful insertion of bolt into plate
            reward = 4
        return reward


class CoordinatedScrewingTask(BimanualViperXTask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 7  # More stages for the coordinated task
        self.nut_placed = False
        self.bolt_placed = False
        self.screwdriver_aligned = False
        self.screwing_started = False

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # Reset task state
        self.nut_placed = False
        self.bolt_placed = False
        self.screwdriver_aligned = False
        self.screwing_started = False

        # Reset physics
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            # Set positions for screwdriver, bolt, and nut (plate is fixed)
            physics.named.data.qpos[-7 * 3 :] = BOX_POSE[0][:21]  # three objects with free joints
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        # Return reward based on task progress
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)

        # Check contacts between objects and grippers
        touch_right_gripper_screwdriver = ("screwdriver", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_left_gripper_bolt = ("bolt", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_left_gripper_nut = ("nut", "vx300s_left/10_left_gripper_finger") in all_contact_pairs

        # Check contacts with plate and table
        nut_touch_plate = ("nut", "plate") in all_contact_pairs
        bolt_touch_nut = ("bolt", "nut") in all_contact_pairs
        screwdriver_touch_bolt = ("screwdriver", "bolt") in all_contact_pairs

        # Check if objects are on the table (failure condition)
        screwdriver_touch_table = ("screwdriver", "table") in all_contact_pairs
        bolt_touch_table = ("bolt", "table") in all_contact_pairs
        nut_touch_table = ("nut", "table") in all_contact_pairs

        # Check if bolt is in plate hole
        bolt_in_plate = ("bolt", "plate_hole") in all_contact_pairs

        # Get positions for checking alignment
        nut_pos = physics.named.data.xpos['nut']
        bolt_pos = physics.named.data.xpos['bolt']
        screwdriver_pos = physics.named.data.xpos['screwdriver']
        plate_pos = physics.named.data.xpos['plate']

        # Check if nut is at center of plate (approximately)
        nut_at_center = np.linalg.norm(nut_pos[:2] - plate_pos[:2]) < 0.03

        # Check if bolt is above nut and perpendicular
        bolt_above_nut = (bolt_pos[2] > nut_pos[2]) and (np.linalg.norm(bolt_pos[:2] - nut_pos[:2]) < 0.02)

        # Check if screwdriver is aligned with bolt
        screwdriver_aligned_with_bolt = (np.linalg.norm(screwdriver_pos[:2] - bolt_pos[:2]) < 0.02)

        # Define reward stages for the coordinated task
        reward = 0

        # Stage 1: Left arm grabs nut, right arm grabs screwdriver
        if touch_left_gripper_nut and touch_right_gripper_screwdriver:
            reward = 1

        # Stage 2: Nut placed flat at center of plate
        if nut_at_center and nut_touch_plate and not self.nut_placed:
            self.nut_placed = True
            reward = 2

        # Stage 3: Left arm grabs bolt after placing nut
        if self.nut_placed and touch_left_gripper_bolt:
            reward = 3

        # Stage 4: Bolt positioned perpendicular to nut
        if self.nut_placed and bolt_above_nut and bolt_touch_nut and not self.bolt_placed:
            self.bolt_placed = True
            reward = 4

        # Stage 5: Right arm positions screwdriver above bolt
        if self.bolt_placed and screwdriver_aligned_with_bolt and not self.screwdriver_aligned:
            self.screwdriver_aligned = True
            reward = 5

        # Stage 6: Screwdriver contacts bolt head
        if self.screwdriver_aligned and screwdriver_touch_bolt and not self.screwing_started:
            self.screwing_started = True
            reward = 6

        # Stage 7: Successful screwing (bolt in plate hole)
        if self.screwing_started and bolt_in_plate:
            reward = 7

        # Failure conditions (objects dropped on table)
        if screwdriver_touch_table or bolt_touch_table or nut_touch_table:
            reward = 0

        return reward
