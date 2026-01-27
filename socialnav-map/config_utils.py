import habitat
import falcon
from habitat.config.default_structured_configs import register_hydra_plugin

from habitat_baselines.config.default_structured_configs import (
    HabitatBaselinesConfigPlugin,
)
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
import hydra
class HabitatConfigPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.append(provider="evalai", path="input/")

register_hydra_plugin(HabitatConfigPlugin)
register_hydra_plugin(HabitatBaselinesConfigPlugin)
@hydra.main(
    version_base=None,
    config_path="config",
    config_name="pointnav/ppo_pointnav_example",
)
def hydra_main_wrapper(cfg):
    """如果需要从 Hydra 配置启动，可以使用这个函数"""
    return cfg

from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
HM3D_CONFIG_PATH = "<YOUR SAVE PATH>/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
MP3D_CONFIG_PATH = "<YOUR SAVE PATH>/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
R2R_CONFIG_PATH = "/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/habitat-lab-0.3.1/habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml"
RXR_CONFIG_PATH = "/home/ubuntu/socialnav/InstructNav_r2r_lf/falcon_hm3d.yaml"
# 添加 PointNav 配置路径
POINTNAV_CONFIG_PATH = "/home/ubuntu/VLA/socialnav-map2/socialnav-map/Falcon/habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_hm3d.yaml"
POINTNAV_CONFIG_MP3D_PATH = "/home/ubuntu/VLA/socialnav-map2/socialnav-map/Falcon/habitat-baselines/habitat_baselines/config/social_nav_v2/falcon_mp3d.yaml"

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/home/PJLAB/caiwenzhe/Desktop/dataset/scenes"
        habitat_config.habitat.dataset.data_path = "/home/PJLAB/caiwenzhe/Desktop/dataset/habitat_task/objectnav/hm3d/v2/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/home/PJLAB/caiwenzhe/Desktop/dataset/scenes/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=90,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config
    
def mp3d_config(path:str=MP3D_CONFIG_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/home/PJLAB/caiwenzhe/Desktop/dataset/scenes"
        habitat_config.habitat.dataset.data_path = "/home/PJLAB/caiwenzhe/Desktop/dataset/habitat_task/objectnav/mp3d/v1/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/home/PJLAB/caiwenzhe/Desktop/dataset/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config

def r2r_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        # habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d"
        habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets"
        habitat_config.habitat.dataset.data_path = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        # habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })  
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3
    return habitat_config

def rxr_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        # habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d"
        habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets"
        habitat_config.habitat.dataset.data_path = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        # habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })  
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3
    return habitat_config


def pointnav_config(path:str=POINTNAV_CONFIG_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        # habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d"
        habitat_config.habitat.dataset.scenes_dir = "/home/ubuntu/VLA/matterport/data/scene_datasets"
        # habitat_config.habitat.dataset.data_path = "/home/ubuntu/socialnav/Falcon/data/datasets/pointnav/social-hm3d/{split}/{split}.json.gz"
        # habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.scene_dataset = "/home/ubuntu/VLA/matterport/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            # "collisions": CollisionsMeasurementConfig(),
        })  
    return habitat_config


def pointnav_config_mp3d(path:str=POINTNAV_CONFIG_MP3D_PATH,stage:str='val',episodes=200):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        # habitat_config.habitat.dataset.scenes_dir = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d"
        habitat_config.habitat.dataset.scenes_dir = "/home/ubuntu/mp3d/v0.1/v1/tasks"
        # habitat_config.habitat.dataset.data_path = "/home/ubuntu/socialnav/Falcon/data/datasets/pointnav/social-hm3d/{split}/{split}.json.gz"
        # habitat_config.habitat.simulator.scene_dataset = "/mnt/hpfs/baaiei/habitat/VLN-CE/data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.scene_dataset = "/home/ubuntu/mp3d/v0.1/v1/tasks/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.environment.iterator_options.num_episode_sample = episodes
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            # "collisions": CollisionsMeasurementConfig(),
        })  



        # print(habitat_config.habitat.task.measurements.keys())
        # exit()
        # habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        # habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        

        
    
    return habitat_config
