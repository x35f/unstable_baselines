import os.path as osp

import gym
import numpy as np
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.utils.config import Config
#from egpo_utils.common import expert_action_prob, ExpertObservation
from metadrive.utils import clip
import math
from metadrive.obs.observation_base import ObservationBase

def expert_action_prob(action, obs, weights, deterministic=False):
    obs = obs.reshape(1, -1)
    x = np.matmul(obs, weights["default_policy/fc_1/kernel"]) + weights["default_policy/fc_1/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_2/kernel"]) + weights["default_policy/fc_2/bias"]
    x = np.tanh(x)
    x = np.matmul(x, weights["default_policy/fc_out/kernel"]) + weights["default_policy/fc_out/bias"]
    x = x.reshape(-1)
    mean, log_std = np.split(x, 2)
    std = np.exp(log_std)
    a_0_p = normpdf(action[0], mean[0], std[0])
    a_1_p = normpdf(action[1], mean[1], std[1])
    expert_action = np.random.normal(mean, std) if not deterministic else mean
    return expert_action, a_0_p, a_1_p

class StateObservation(ObservationBase):
    def __init__(self, config):
        super(StateObservation, self).__init__(config)

    @property
    def observation_space(self):
        # Navi info + Other states
        shape = 19
        return gym.spaces.Box(-0.0, 1.0, shape=(shape,), dtype=np.float32)

    def observe(self, vehicle):
        navi_info = vehicle.navigation.get_navi_info()
        ego_state = self.vehicle_state(vehicle)
        ret = np.concatenate([ego_state, navi_info])
        return ret.astype(np.float32)

    def vehicle_state(self, vehicle):
        # update out of road
        current_reference_lane = vehicle.navigation.current_ref_lanes[-1]
        lateral_to_left, lateral_to_right = vehicle.dist_to_left_side, vehicle.dist_to_right_side
        total_width = float(
            (vehicle.navigation.map.config["lane_num"] + 1) * vehicle.navigation.map.config["lane_width"]
        )
        info = [
            clip(lateral_to_left / total_width, 0.0, 1.0),
            clip(lateral_to_right / total_width, 0.0, 1.0),
            vehicle.heading_diff(current_reference_lane),
            # Note: speed can be negative denoting free fall. This happen when emergency brake.
            clip((vehicle.speed + 1) / (vehicle.max_speed + 1), 0.0, 1.0),
            clip((vehicle.steering / vehicle.max_steering + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][0] + 1) / 2, 0.0, 1.0),
            clip((vehicle.last_current_action[0][1] + 1) / 2, 0.0, 1.0)
        ]
        heading_dir_last = vehicle.last_heading_dir
        heading_dir_now = vehicle.heading
        cos_beta = heading_dir_now.dot(heading_dir_last
                                       ) / (np.linalg.norm(heading_dir_now) * np.linalg.norm(heading_dir_last))

        beta_diff = np.arccos(clip(cos_beta, 0.0, 1.0))

        # print(beta)
        yaw_rate = beta_diff / 0.1
        # print(yaw_rate)
        info.append(clip(yaw_rate, 0.0, 1.0))
        _, lateral = vehicle.lane.local_coordinates(vehicle.position)
        info.append(clip((lateral * 2 / vehicle.navigation.map.config["lane_width"] + 1.0) / 2.0, 0.0, 1.0))
        return info

class ExpertObservation(ObservationBase):
    def __init__(self, vehicle_config):
        self.state_obs = StateObservation(vehicle_config)
        super(ExpertObservation, self).__init__(vehicle_config)
        self.cloud_points = None
        self.detected_objects = None

    @property
    def observation_space(self):
        shape = list(self.state_obs.observation_space.shape)
        if self.config["lidar"]["num_lasers"] > 0 and self.config["lidar"]["distance"] > 0:
            # Number of lidar rays and distance should be positive!
            shape[0] += self.config["lidar"]["num_lasers"] + self.config["lidar"]["num_others"] * 4
        return gym.spaces.Box(-0.0, 1.0, shape=tuple(shape), dtype=np.float32)

    def observe(self, vehicle):
        state = self.state_observe(vehicle)
        other_v_info = self.lidar_observe(vehicle)
        self.current_observation = np.concatenate((state, np.asarray(other_v_info)))
        ret = self.current_observation
        return ret.astype(np.float32)

    def state_observe(self, vehicle):
        return self.state_obs.observe(vehicle)

    def lidar_observe(self, vehicle):
        other_v_info = []
        if vehicle.lidar.available:
            cloud_points, detected_objects = vehicle.lidar.perceive(vehicle, )
            other_v_info += vehicle.lidar.get_surrounding_vehicles_info(
                vehicle, detected_objects, 4)
            other_v_info += cloud_points
            self.cloud_points = cloud_points
            self.detected_objects = detected_objects
        return other_v_info

    
def normpdf(x, mean, sd):
    var = float(sd) ** 2
    denom = (2 * math.pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def load_weights(path: str):
    """
    Load NN weights
    :param path: weights file path path
    :return: NN weights object
    """
    # try:
    model = np.load(path)
    return model
    # except FileNotFoundError:
    # print("Can not find {}, didn't load anything".format(path))
    # return None


class ExpertGuidedEnv(SafeMetaDriveEnv):

    def default_config(self) -> Config:
        """
        Train/Test set both contain 10 maps
        :return: PGConfig
        """
        config = super(ExpertGuidedEnv, self).default_config()
        config.update(dict(
            environment_num=100,
            start_seed=100,
            safe_rl_env_v2=False,  # If True, then DO NOT done even out of the road!
            # _disable_detector_mask=True,  # default False to acc Lidar detection

            # traffic setting
            random_traffic=False,
            # traffic_density=0.1,

            # special setting
            rule_takeover=False,
            takeover_cost=1,
            cost_info="native",  # or takeover
            random_spawn=False,  # used to collect dataset
            cost_to_reward=True,  # for egpo, it accesses the ENV reward by penalty
            horizon=1000,

            crash_vehicle_penalty=1.,
            crash_object_penalty=0.5,
            out_of_road_penalty=1.,

            vehicle_config=dict(  # saver config, free_level:0 = expert
                use_saver=False,
                free_level=100,
                expert_deterministic=False,
                release_threshold=100,  # the save will be released when level < this threshold
                overtake_stat=False),  # set to True only when evaluate

            expert_value_weights=osp.join(osp.dirname(__file__), "expert.npz")
        ), allow_add_new_key=True)
        return config

    def __init__(self, config):
        # if ("safe_rl_env" in config) and (not config["safe_rl_env"]):
        #     raise ValueError("You should always set safe_rl_env to True!")
        # config["safe_rl_env"] = True
        if config.get("safe_rl_env_v2", False):
            config["out_of_road_penalty"] = 0
        super(ExpertGuidedEnv, self).__init__(config)
        self.expert_observation = ExpertObservation(self.config["vehicle_config"])
        assert self.config["expert_value_weights"] is not None
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        self.state_value = 0
        self.expert_weights = load_weights(self.config["expert_value_weights"])
        if self.config["cost_to_reward"]:
            self.config["out_of_road_penalty"] = self.config["out_of_road_cost"]
            self.config["crash_vehicle_penalty"] = self.config["crash_vehicle_cost"]
            self.config["crash_object_penalty"] = self.config["crash_object_cost"]

    def expert_observe(self):
        return self.expert_observation.observe(self.vehicle)

    def get_expert_action(self):
        obs = self.expert_observation.observe(self.vehicle)
        return expert_action_prob([0, 0], obs, self.expert_weights,
                                  deterministic=False)[0]

    def _get_reset_return(self):
        assert self.num_agents == 1
        self.total_takeover_cost = 0
        self.total_native_cost = 0
        if self.config["vehicle_config"]["free_level"] < 1e-3:
            # 1.0 full takeover
            self.vehicle.takeover_start = True
        return super(ExpertGuidedEnv, self)._get_reset_return()

    def step(self, actions):
        actions, saver_info = self.expert_takeover("default_agent", actions)
        obs, r, d, info, = super(ExpertGuidedEnv, self).step(actions)
        saver_info.update(info)
        info = self.extra_step_info(saver_info)
        return obs, r, d, info

    def extra_step_info(self, step_info):
        # step_info = step_infos[self.DEFAULT_AGENT]

        step_info["native_cost"] = step_info["cost"]
        # if step_info["out_of_road"] and not step_info["arrive_dest"]:
        # out of road will be done now
        step_info["high_speed"] = True if self.vehicle.speed >= 50 else False
        step_info["takeover_cost"] = self.config["takeover_cost"] if step_info["takeover_start"] else 0
        self.total_takeover_cost += step_info["takeover_cost"]
        self.total_native_cost += step_info["native_cost"]
        step_info["total_takeover_cost"] = self.total_takeover_cost
        step_info["total_native_cost"] = self.total_native_cost

        if self.config["cost_info"] == "native":
            step_info["cost"] = step_info["native_cost"]
            step_info["total_cost"] = self.total_native_cost
        elif self.config["cost_info"] == "takeover":
            step_info["cost"] = step_info["takeover_cost"]
            step_info["total_cost"] = self.total_takeover_cost
        else:
            raise ValueError
        return step_info

    def done_function(self, v_id):
        """This function is a little bit different compared to the SafePGDriveEnv in PGDrive!"""
        done, done_info = super(ExpertGuidedEnv, self).done_function(v_id)
        if self.config["safe_rl_env_v2"]:
            assert self.config["out_of_road_cost"] > 0
            if done_info["out_of_road"]:
                done = False
        return done, done_info

    def _is_out_of_road(self, vehicle):
        return vehicle.out_of_route

    def expert_takeover(self, v_id: str, actions):
        """
        Action prob takeover
        """
        if self.config["rule_takeover"]:
            return self.rule_takeover(v_id, actions)
        vehicle = self.vehicles[v_id]
        action = actions
        steering = action[0]
        throttle = action[1]
        self.state_value = 0
        pre_save = vehicle.takeover
        if vehicle.config["use_saver"] or vehicle.expert_takeover:
            # saver can be used for human or another AI
            free_level = vehicle.config["free_level"] if not vehicle.expert_takeover else 1.0
            obs = self.expert_observation.observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_weights,
                                                           deterministic=vehicle.config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
                saver_a = action
            else:
                if free_level <= 1e-3:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif free_level > 1e-3:
                    if a_0_p * a_1_p < 1 - vehicle.config["free_level"]:
                        steering, throttle = saver_a[0], saver_a[1]

        # indicate if current frame is takeover step
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        if saver_info["takeover"]:
            saver_info["raw_action"] = [steering, throttle]
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

    def rule_takeover(self, v_id, actions):
        vehicle = self.vehicles[v_id]
        action = actions[v_id]
        steering = action[0]
        throttle = action[1]
        if vehicle.config["use_saver"] or vehicle.expert_takeover:
            # saver can be used for human or another AI
            save_level = vehicle.config["save_level"] if not vehicle.expert_takeover else 1.0
            obs = self.observations[v_id].observe(vehicle)
            try:
                saver_a, a_0_p, a_1_p = expert_action_prob(action, obs, self.expert_weights,
                                                           deterministic=vehicle.config["expert_deterministic"])
            except ValueError:
                print("Expert can not takeover, due to observation space mismathing!")
            else:
                if save_level > 0.9:
                    steering = saver_a[0]
                    throttle = saver_a[1]
                elif save_level > 1e-3:
                    heading_diff = vehicle.heading_diff(vehicle.lane) - 0.5
                    f = min(1 + abs(heading_diff) * vehicle.speed * vehicle.max_speed, save_level * 10)
                    # for out of road
                    if (obs[0] < 0.04 * f and heading_diff < 0) or (obs[1] < 0.04 * f and heading_diff > 0) or obs[
                        0] <= 1e-3 or \
                            obs[
                                1] <= 1e-3:
                        steering = saver_a[0]
                        throttle = saver_a[1]
                        if vehicle.speed < 5:
                            throttle = 0.5
                    # if saver_a[1] * vehicle.speed < -40 and action[1] > 0:
                    #     throttle = saver_a[1]

                    # for collision
                    lidar_p = vehicle.lidar.get_cloud_points()
                    left = int(vehicle.lidar.num_lasers / 4)
                    right = int(vehicle.lidar.num_lasers / 4 * 3)
                    if min(lidar_p[left - 4:left + 6]) < (save_level + 0.1) / 10 or min(lidar_p[right - 4:right + 6]
                                                                                        ) < (save_level + 0.1) / 10:
                        # lateral safe distance 2.0m
                        steering = saver_a[0]
                    if action[1] >= 0 and saver_a[1] <= 0 and min(min(lidar_p[0:10]), min(lidar_p[-10:])) < save_level:
                        # longitude safe distance 15 m
                        throttle = saver_a[1]

        # indicate if current frame is takeover step
        pre_save = vehicle.takeover
        vehicle.takeover = True if action[0] != steering or action[1] != throttle else False
        saver_info = {
            "takeover_start": True if not pre_save and vehicle.takeover else False,
            "takeover_end": True if pre_save and not vehicle.takeover else False,
            "takeover": vehicle.takeover if pre_save else False
        }
        return (steering, throttle) if saver_info["takeover"] else action, saver_info

if __name__ == "__main__":

    env = ExpertGuidedEnv(dict(
        vehicle_config=dict(
            use_saver=True,
            free_level=0.95,
            # overtake_stat=True,
            expert_deterministic=False,
            # spawn_lateral=7,
            # increment_steering=True
            show_navi_mark=False
          ),
        # camera_height=7,
        # accident_prob=1.0,
        # traffic_density=0.3,
        # traffic_mode="respawn",
        cull_scene=True,
        map_config={
            "config": 5,
        },
        # camera_dist=10,
        cost_to_reward=True,
        # crash_vehicle_penalty=1.,
        # crash_object_penalty=0.5,
        # out_of_road_penalty=1.,
        # crash_object_penalty=5,
        # crash_vehicle_penalty=5,
        # rule_takeover=True,

        safe_rl_env=True,
        use_render=True,
        # debug=True,
        manual_control=False))

    def _save(env):
        env.vehicle.vehicle_config["use_saver"]= not env.vehicle.vehicle_config["use_saver"]

    eval_reward = []
    done_num=0
    o = env.reset()
    # env.vehicle.remove_display_region()
    env.main_camera.set_follow_lane(True)
    env.engine.accept("p",env.capture)
    env.engine.accept("u", _save, extraArgs=[env])
    max_s = 0
    max_t = 0
    start = 0
    total_r = 0
    for i in range(1, 30000):
        o_to_evaluate = o
        o, r, d, info = env.step(env.action_space.sample())
        total_r += r
        max_s = max(max_s, info["raw_action"][0])
        max_t = max(max_t, info["raw_action"][1])

        # assert not info["takeover_start"]
        text = {
                # "save": env.vehicle.takeover, "overtake_num": info["overtake_vehicle_num"],
                # "native_cost": info["native_cost"], "total_native_cost": info["total_native_cost"],
                "reward": total_r, "takeover_cost": info["takeover_cost"],
                # "total_takeover_cost": info["total_takeover_cost"],
                # "takeover start": info["takeover_start"], "takeover end": info["takeover_end"],
                "Takeover": info["takeover"],
                # "raw_action": env.vehicle.last_current_action[1],
                # "state_value": env.state_value,
                # "map_seed": env.current_map.random_seed,
                "Cost":int(info["total_native_cost"]),
                # "total_cost":info["total_cost"],
                # "crash_vehicle":info["crash_vehicle"],
                # "crash_object":info["crash_object"]
                # "current_map":env.current_map.random_seed
        }
        if env.config["cost_info"] == "native":
            assert info["cost"] == info["native_cost"]
            assert info["total_cost"] == info["total_native_cost"]
        elif env.config["cost_info"] == "takeover":
            assert info["cost"] == info["takeover_cost"]
            assert info["total_cost"] == info["total_takeover_cost"]
        else:
            raise ValueError
        # if info["takeover_start"] and not env.config["manual_control"]:
        #     print(info["raw_action"])
        #     assert info["raw_action"] == (0,1)
        #     assert not info["takeover"]
        # if info["takeover"] and not env.config["manual_control"]:
        #     print(info["raw_action"])
        #     assert info["raw_action"] != (0,1)
        # print(r)
        env.render(text=text)
        if d:
            eval_reward.append(total_r)
            done_num+=1
            if done_num > 100:
                break
            print(info["out_of_road"])
            print("done_cost:{}".format(info["cost"]))
            print("done_reward:{}".format(r))
            print("total_takeover_cost:{}".format(info["total_takeover_cost"]))
            takeover_cost = 0
            native_cost = 0
            total_r = 0
            print("episode_len:", i - start)
            env.reset()
            start = i
    import numpy as np
    print(np.mean(eval_reward),np.std(sorted(eval_reward)))
    env.close()