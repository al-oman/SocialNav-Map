import habitat
import numpy as np
import cv2
import ast
import torch
import sys
import math
from typing import TYPE_CHECKING, Union, cast
from HumanTrajectoryPredictor import HumanTrajectoryPredictor
import time
import quaternion
from habitat.utils.visualizations import maps
sys.path.append('/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf')
# from utils.semantic_prediction import SemanticPredMaskRCNN
# from llava_infer import LLaVANavigationinference1
from huatu3 import process_semantic_map
import pickle
from torchvision import transforms
from PIL import Image
import skimage.morphology
from constants import color_palette
import utils.pose as pu
from utils.fmm_planner import FMMPlanner
import utils.visualization as vu
import os
import shutil
# from RedNet.RedNet_model import load_rednet
from constants import mp_categories_mapping
import torch

from skimage import measure



def create_action_args_dict(action):
    """
    创建action_args字典
    key是agent action名称，value是对应的numpy数组
    """
    # 根据action确定第一个agent的action名称
    agent_0_actions = {
        0: 'agent_0_discrete_stop',
        1: 'agent_0_discrete_move_forward',
        2: 'agent_0_discrete_turn_left',
        3: 'agent_0_discrete_turn_right'
    }
    
    # 创建action_args字典
    action_args_dict = {
        agent_0_actions[action]: np.array([action], dtype=np.int64),
        'agent_1_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64),
        'agent_2_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64),
        'agent_3_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64),
        'agent_4_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64),
        'agent_5_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64),
        'agent_6_oracle_nav_randcoord_action_obstacle': np.array([], dtype=np.int64)
    }
    
    return action_args_dict


class UnTrapHelper:
    def __init__(self):
        self.total_id = 0
        self.epi_id = 0
    
    def reset(self):
        self.total_id +=1
        self.epi_id = 0
    
    def get_action(self):
        self.epi_id +=1
        if self.epi_id ==1:
            if self.total_id %2==0:
                return 2
            else:
                return 3
        else:
            if self.total_id %2==0:
                return 2
            else:
                return 3

class HM3D_Objnav_Agent:
    def __init__(self,env:habitat.Env,args,rank=0):
        self.env = env
        self.rank = rank
        # 获取底层的simulator
        sim = env.sim
        print(f"Simulator type: {type(sim)}")
        # exit()
        self.args = args
        self.episode_samples = 0
        # self.planner = ShortestPathFollower(env.sim,0.5,False)
        self.cnt = 1
        self.visited_vis = None
        self.visited = None
        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((args.frame_height, args.frame_width),
                               interpolation=Image.NEAREST)])
        
        self.device = args.device
        # self.red_sem_pred = load_rednet(
        #     self.device, ckpt='RedNet/model/rednet_semmap_mp3d_40.pth', resize=True, # since we train on half-vision
        # )
        # self.red_sem_pred.eval()
        self.curr_loc = None
        self.last_loc = None
        if args.visualize or args.print_images:
            self.legend = cv2.imread('docs/legend.png')
            self.vis_image = None
            self.rgb_vis = None
        self.goal_name = 'test'
        self.timestep = 0
        model = args.model_dir
        self.actionlist = []
        self.dir = "{}/dump/{}/episodes".format(args.dump_location,args.exp_name)
        self.poselist = []
        self.last_action = None
        self.collision_n = 0
        self.untrap = UnTrapHelper()
        # print(self.env.get_metrics())
        # exit()
        self.orcale_success = 0

        # 添加规划相关属性
        self.selem = skimage.morphology.disk(3)
        self.collision_map = None
        self.col_width = 1
        self.replan_count = 0
        self.count_forward_actions = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.goal =None
        self.goal_world_pos = None  # 存储目标的世界坐标
        self.goal_set = False       # 标记目标是否已设置

        # 添加目标相关属性
        self.goal_world_pos = None  # 存储目标的世界坐标
        self.goal_map_coords = None  # 存储目标在地图中的坐标
        self.goal = None  # 地图中的goal矩阵

        self.goal_global_map_pos = None

        # 基础inference1路径
        self.base_inference1_path = f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}'
        self.episode_no=0

        # 添加动态障碍物相关属性
        self.human_obstacle_memory = {}  # 存储人类经过位置的时间戳
        self.obstacle_decay_time = 5  # 障碍物衰减时间（步数）
        self.obstacle_radius = 0.25  # 人类影响半径
        self.same_height_threshold = 0.1


        # 添加固定参考系
        self.initial_agent_position = None
        self.initial_agent_rotation = None


        # 添加人类影响过的所有区域记录
        self.human_affected_areas = set()


        # 添加轨迹预测模块
        self.trajectory_predictor = HumanTrajectoryPredictor(
            history_length=10,      # 使用最近5个位置点进行拟合
            prediction_steps=10,   # 预测未来10步
            prediction_interval=10  # 每3步更新一次预测
        )

        # 存储预测的障碍物
        self.predicted_obstacles = {}
        self.predicted_obstacle_decay_time = 5  # 预测障碍物的衰减时间
        # 添加调试相关属性
        self.debug_log = []
        self.human_trajectory_log = {}

        # 添加基于朝向的直线预测参数 - 新增部分
        self.prediction_distance = 0.5        # 预测距离（米）
        self.prediction_width = 0.25           # 预测宽度（米，朝向两侧各0.25米）
        self.prediction_steps = 10            # 预测步数
        self.width_steps = 5                 # 宽度方向的采样点数
        self.distance_decay_factor = 0.5      # 距离衰减系数（0-1，越大衰减越快）
        self.width_decay_factor = 0.3         # 宽度衰减系数（0-1，越大衰减越快）
        self.min_prediction_weight = 0.1      # 最小预测权重阈值

    def reset(self,no):
        args = self.args
        self.timestep = 0
        self.episode_no =no
        self.curr_loc = [args.map_size_cm / 100.0 / 2.0,
                         args.map_size_cm / 100.0 / 2.0, 0.]
        self.last_loc = None
        self.episode_samples += 1
        self.episode_steps = 0
        self.obs = self.env.reset()
        self.trajectory_summary = ""  
        map_shape = (args.map_size_cm // args.map_resolution,
                     args.map_size_cm // args.map_resolution)
        self.visited = np.zeros(map_shape)
        self.visited_vis = np.zeros(map_shape)
        if args.visualize or args.print_images:
            self.vis_image = vu.init_vis_image(self.goal_name, self.legend)


        # print("所有观察键:", list(self.obs.keys()))
        # print("所有episode:", list(self.env.current_episode.goals))
        
        
        # print(self.env.current_episode.goals[0].position,self.env.sim.get_agent_state())
        
        # for i in range(7):
        #     print(self.env.sim.get_agent_state(i))
        # # print(self.env.sim._sensors)
        # exit()
        
        self.goal_world_pos = self.env.current_episode.goals[0].position
        
        # exit()
        rgb = self.obs['agent_0_articulated_agent_jaw_rgb'].astype(np.uint8)
        
        # cv2.imwrite('rgb.png',rgb)
        # print(np.min(self.depth),np.max(self.depth))
        depth = (self.obs['agent_0_articulated_agent_jaw_depth'] - 0.5) / 4.5
        # print(depth.shape)
        # print(np.min(depth),np.max(depth))
        cv2.imwrite('depth.png',depth*255)
        # exit()
        # depth = depth[:, :, None]  # 将深度图像从 (224, 224) 转换为 (224, 224, 1)
        # depth = depth / 255.0
        semantic = torch.zeros((rgb.shape[0], rgb.shape[1],1))
        state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)

        obs = self._preprocess_obs(state)
        obs = np.expand_dims(obs, axis=0)
        obs = torch.from_numpy(obs).float().to(self.device)
        self.position = self.env.sim.get_agent_state().position
        self.rotation = self.env.sim.get_agent_state().rotation


        # 保存初始参考系 - 关键添加
        self.initial_agent_position = self.position
        self.initial_agent_rotation = self.rotation


        # 计算目标在全局地图中的固定位置
        self.goal_global_map_pos = self._world_to_global_map_coords(
            self.goal_world_pos, self.position, self.rotation)
        # print(self.position,self.rotation)
        # exit()
        self.actionlist = []
        self.dir = "{}/dump/{}/episodes".format(args.dump_location,args.exp_name)
        self.poselist = []
        self.last_action = None
        self.collision_n = 0
        self.orcale_success = 0

        # 添加碰撞地图初始化
        self.collision_map = np.zeros(map_shape)
        self.col_width = 1
        self.replan_count = 0
        self.count_forward_actions = 0


        # self.goal_world_pos = None  # 存储目标的世界坐标
        self.goal_set = False       # 标记目标是否已设置
        self.metrics = None
        self.zaizouyibu = 0
        self.episode_folder = os.path.join(
                self.base_inference1_path, 
                f'rank_{self.episode_no}'
            )
            
        # 创建文件夹（如果不存在）
        os.makedirs(self.episode_folder, exist_ok=True)
        
        self.human_obstacle_memory = {}

        # 添加这两行：
        self.predicted_obstacles = {}
        self.trajectory_predictor.reset()  # 添加这行


        return obs,self.position,self.rotation
    
    def get_topdownmap(self):
        topdownmap = self.metrics['top_down_map']
        # print(self.metrics['top_down_map'])
        # cv2.imwrite('topdownmap.png',topdownmap['map'])
        unique_values = np.unique(topdownmap['map'])
        # print(f"Map unique values: {unique_values}")
        semantic_colors = {
            0: (0, 0, 0),         # 黑色 - 未知/无效区域
            1: (255, 255, 255),   # 白色 - 可通行区域/地面
            2: (128, 128, 128),   # 灰色 - 墙壁/障碍物
            4: (0, 255, 0),       # 绿色 - 可能是植物/户外
            6: (255, 0, 0),       # 红色 - 家具/物体
            14: (0, 255, 255),    # 青色 - 门/窗户
            24: (255, 255, 0),    # 黄色 - 特殊区域
            34: (255, 0, 255),    # 品红 - 装饰品
            44: (0, 128, 255),    # 橙蓝 - 其他物体
        }

        # 为每个语义类别着色
        # 创建RGB图像
        height, width = topdownmap['map'].shape
        colored_map = np.zeros((height, width, 3), dtype=np.uint8)


        for value in unique_values:
            if value in semantic_colors:
                mask = topdownmap['map'] == value
                colored_map[mask] = semantic_colors[value]

        return colored_map
    
    def update_trajectory(self):
        self.episode_steps += 1
        self.metrics = self.env.get_metrics()
        # print(self.metrics)
        # exit()
        

        
        # exit()
        if self.metrics['distance_to_goal'] <=3:
            self.orcale_success = 1
        # self.rgb_trajectory.append(cv2.cvtColor(self.obs['agent_0_articulated_agent_jaw_rgb'],cv2.COLOR_BGR2RGB))
        # self.depth_trajectory.append((self.obs['agent_0_articulated_agent_jaw_depth']/5.0 * 255.0).astype(np.uint8))
        
        self.position = self.env.sim.get_agent_state().position
        self.rotation = self.env.sim.get_agent_state().rotation

        # cv2.imwrite("monitor-rgb.jpg",self.obs['agent_0_articulated_agent_jaw_rgb'])
        # cv2.imwrite("monitor-depth.jpg",self.obs['agent_0_articulated_agent_jaw_depth']/5.0 * 255.0)
            
    
    def step(self,planner_inputs):
        args = self.args


        # timebegin = time.time()
        # if self.timestep<=3:

        #     action = 1
        # else:
        #     action = 0
        if not self.env.episode_over:
            action = self._plan(planner_inputs)

            # print(action)
            # exit()
            if self.args.visualize or self.args.print_images:
                self._visualize(planner_inputs)


            
            
            # print(f"Created episode folder: {self.episode_folder}")
            # cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/rgb{self.rank}-{args.exp_name}-{self.timestep}.png',self.obs['agent_0_articulated_agent_jaw_rgb'][:,:,::-1])
            if self.args.collect == 1:
                cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/rank_{self.episode_no}/jaw_rgb{self.rank}-{args.exp_name}-{self.timestep}.png',self.obs['agent_0_articulated_agent_jaw_rgb'][:,:,::-1])
                cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/rank_{self.episode_no}/third_rgb{self.rank}-{args.exp_name}-{self.timestep}.png',self.obs['agent_0_articulated_agent_arm_rgb'][:,:,::-1])
                cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/rank_{self.episode_no}/jaw_depth{self.rank}-{args.exp_name}-{self.timestep}.png',(self.obs['agent_0_articulated_agent_jaw_depth'] - 0.5) / 4.5 * 255)
                cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/rank_{self.episode_no}/sem{self.rank}-{args.exp_name}-{self.timestep}.png',self.vis_image[50:530, 670:1150])
            # exit()
            # labeled_image,labels,objects = process_semantic_map(f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference1/sem{self.rank}-{args.exp_name}.png', f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference1/sem_labels{self.rank}-{args.exp_name}.png')
            # images= [f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference1/rgb{self.rank}-{args.exp_name}-{self.timestep}.png',f'/mnt/hpfs/baaiei/habitat/InstructNav_r2r_lf/inference1/sem_labels{self.rank}-{args.exp_name}.png']


            # end = time.time()
            # print("time:",start-end)
            # stop
            if self.timestep>args.stop_th:
                action = 0
            # self.actionlist.append(action)
            if action==1:
                self.actionlist.append(f'Step {self.timestep}: MOVE_FORWARD\n')
            elif action==2:
                self.actionlist.append(f'Step {self.timestep}: TURN_LEFT\n')
            elif action==3:
                self.actionlist.append(f'Step {self.timestep}: TURN_RIGHT\n')

            # print(self.actionlist)
            untrap = 0
            if self.collision_n >= 6 and untrap==1 and action!=0:
                action =self.untrap.get_action()


            # print(self.env.get_metrics())
            # print(action)

            # 第一个agent的action映射
            first_agent_actions = {
                0: 'agent_0_discrete_stop',
                1: 'agent_0_discrete_move_forward',
                2: 'agent_0_discrete_turn_left',
                3: 'agent_0_discrete_turn_right'
            }
            
            # 固定的后6个元素
            fixed_actions = (
                'agent_1_oracle_nav_randcoord_action_obstacle',
                'agent_2_oracle_nav_randcoord_action_obstacle', 
                'agent_3_oracle_nav_randcoord_action_obstacle',
                'agent_4_oracle_nav_randcoord_action_obstacle',
                'agent_5_oracle_nav_randcoord_action_obstacle',
                'agent_6_oracle_nav_randcoord_action_obstacle'
            )
            
            # 组合成完整的tuple

            action_args = create_action_args_dict(action)
            # print(action_args)
            # exit()
            

            action_tuple = (first_agent_actions[action],) + fixed_actions


            action_dict = {'action': action_tuple, 'action_args': action_args}

            # print(action_dict)
            # exit()


            # if self.episode_no<=678:
            #     action = 0


            self.obs = self.env.step(action_dict)

            self.last_action  = action
            # print('type(self.obs):', self.obs)
            # print('type(self.obs):', self.obs['agent_0_articulated_agent_jaw_rgb'].shape) # (256, 256, 3)
            # print('type(self.obs):', self.obs['agent_0_articulated_agent_jaw_depth'].shape) # (256, 256, 1)
            
            
            rgb = self.obs['agent_0_articulated_agent_jaw_rgb'].astype(np.uint8)
            # depth = self.obs['agent_0_articulated_agent_jaw_depth']
            depth = (self.obs['agent_0_articulated_agent_jaw_depth'] - 0.5) / 4.5
            # depth = depth[:, :, None]  # 将深度图像从 (224, 224) 转换为 (224, 224, 1)
            semantic = torch.zeros((rgb.shape[0], rgb.shape[1],1))
            state = np.concatenate((rgb, depth, semantic), axis=2).transpose(2, 0, 1)
            obs = self._preprocess_obs(state)
            obs = np.expand_dims(obs, axis=0)
            obs = torch.from_numpy(obs).float().to(self.device)
            # self.semantic = torch.zeros(1, 16, rgb.shape[0], rgb.shape[1])
            self.update_trajectory()
            self.poselist.append(planner_inputs['sensor_pose'].cpu().numpy().tolist())
            # print(self.poselist)

            # 收集数据
            # collect = 
                

                    # exit()
            self.timestep +=1
            poses = [self.position,self.rotation]
            # print('poses:',poses)
            done = self.env.episode_over

            # if action==0 and self.zaizouyibu == 0:
            #     action = 1
            #     self.zaizouyibu = 1

            # if self.zaizouyibu == 1:
            #     action = 0

            #     print('stop')

            
            if action==0 or self.metrics['human_collision'] == 1.0:
                done =1
                topdownmap = self.get_topdownmap()
                cv2.imwrite(f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/rank_{self.episode_no}/topdownmap.png',topdownmap)
                # 在evaluation_metrics.append之后添加：
                # import json
                # debug_file = f'/home/ubuntu/socialnav/InstructNav_r2r_lf/inference1/{self.args.exp_name}/debug_episode_{self.episode_no}_step_{self.timestep}.json'
                # with open(debug_file, 'w') as f:
                #     json.dump({
                #         'debug_log': self.debug_log[-20:],  # 保存最近20步
                #         'trajectory_log': self.human_trajectory_log
                #     }, f, indent=2)
            if self.episode_no>=11:
                exit()
            # timeend = time.time()
            # print(timeend-timebegin)
            return obs,self.position,self.rotation,done


    def _plan(self, planner_inputs):
        """Function responsible for planning

        Args:
            planner_inputs (dict):
                dict with following keys:
                    'map_pred'  (ndarray): (M, M) map prediction
                    'goal'      (ndarray): (M, M) goal locations
                    'pose_pred' (ndarray): (7,) array  denoting pose (x,y,o)
                                 and planning window (gx1, gx2, gy1, gy2)
                    'found_goal' (bool): whether the goal object is found

        Returns:
            action (int): action id
        """
        args = self.args

        self.last_loc = self.curr_loc

        # Get Map prediction
        original_map_pred = np.rint(planner_inputs['map_pred'])
        map_pred = np.rint(planner_inputs['map_pred'])
        exp_pred = np.rint(planner_inputs['exp_pred'])

        # Get pose prediction and global policy planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = \
            planner_inputs['pose_pred']
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        planning_window = [gx1, gx2, gy1, gy2]


        # 更新人类障碍物记忆
        self.update_human_obstacle_memory(planning_window)

        # 更新轨迹预测（新增）
        self.update_trajectory_prediction(planning_window)

        
        # 应用人类障碍物到地图上（关键修改点）
        map_pred = self.apply_human_obstacles_to_map(original_map_pred, planning_window)


        # 应用预测障碍物到地图上（新增）
        map_pred = self.apply_predicted_obstacles_to_map(map_pred, planning_window)


        # 更新goal地图 - 这是关键修改
        self.goal = self._update_goal_map(map_pred.shape, planner_inputs['pose_pred'])



        # Get curr loc
        self.curr_loc = [start_x, start_y, start_o]
        r, c = start_y, start_x
        start = [int(r * 100.0 / args.map_resolution - gx1),
                 int(c * 100.0 / args.map_resolution - gy1)]
        start = pu.threshold_poses(start, map_pred.shape)

        self.visited[gx1:gx2, gy1:gy2][start[0] - 0:start[0] + 1,
                                       start[1] - 0:start[1] + 1] = 1

        # if args.visualize or args.print_images:
            # Get last loc
        last_start_x, last_start_y = self.last_loc[0], self.last_loc[1]
        r, c = last_start_y, last_start_x
        last_start = [int(r * 100.0 / args.map_resolution - gx1),
                        int(c * 100.0 / args.map_resolution - gy1)]
        last_start = pu.threshold_poses(last_start, map_pred.shape)
        self.visited_vis[gx1:gx2, gy1:gy2] = \
            vu.draw_line(last_start, start,
                            self.visited_vis[gx1:gx2, gy1:gy2])

        # # Collision check
        # if self.last_action == 1 and args.collision:
        #     x1, y1, t1 = self.last_loc
        #     x2, y2, _ = self.curr_loc
        #     buf = 4
        #     length = 2
        
        #     # if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
        #     #     self.col_width += 2
        #     #     if self.col_width == 7:
        #     #         length = 4
        #     #         buf = 3
        #     #     self.col_width = min(self.col_width, 5)
        #     # else:
        #     #     self.col_width = 1

        #     dist = pu.get_l2_distance(x1, x2, y1, y2)
        #     if dist < args.collision_threshold:  # Collision
        #         self.collision_n += 1
        #         # width = self.col_width
        #     else:
        #         self.untrap.reset()

        # Collision check
        if self.last_action == 1:  # 如果上一个动作是前进
            x1, y1, t1 = self.last_loc
            x2, y2, _ = self.curr_loc
            buf = 4
            length = 2

            if abs(x1 - x2) < 0.05 and abs(y1 - y2) < 0.05:
                self.col_width += 2
                if self.col_width == 7:
                    length = 4
                    buf = 3
                self.col_width = min(self.col_width, 5)
            else:
                self.col_width = 1

            dist = pu.get_l2_distance(x1, x2, y1, y2)
            if dist < 0.15:  # 碰撞阈值
                self.collision_n += 1
                width = self.col_width
                for i in range(length):
                    for j in range(width):
                        wx = x1 + 0.05 * \
                            ((i + buf) * np.cos(np.deg2rad(t1))
                            + (j - width // 2) * np.sin(np.deg2rad(t1)))
                        wy = y1 + 0.05 * \
                            ((i + buf) * np.sin(np.deg2rad(t1))
                            - (j - width // 2) * np.cos(np.deg2rad(t1)))
                        r, c = wy, wx
                        r, c = int(r * 100 / args.map_resolution), \
                            int(c * 100 / args.map_resolution)
                        [r, c] = pu.threshold_poses([r, c],
                                                    self.collision_map.shape)
                        self.collision_map[r, c] = 1

        # Get short-term goal
        stg, replan, stop = self._get_stg(map_pred, start, np.copy(self.goal), planning_window)

        if replan:
            self.replan_count += 1
            print("Replan: ", self.replan_count)
        else:
            self.replan_count = 0

        # Deterministic Local Policy
        if stop or self.replan_count > 26:
            action = 0  # Stop
        else:
            (stg_x, stg_y) = stg
            angle_st_goal = math.degrees(math.atan2(stg_x - start[0],
                                                    stg_y - start[1]))
            angle_agent = (start_o) % 360.0
            if angle_agent > 180:
                angle_agent -= 360

            relative_angle = (angle_agent - angle_st_goal) % 360.0
            if relative_angle > 180:
                relative_angle -= 360

            # Simple action selection
            if relative_angle > 15:  # args.turn_angle / 2.
                action = 3  # Right
            elif relative_angle < -15:
                action = 2  # Left
            else:
                action = 1  # Forward
        

        distance_to_goal = np.linalg.norm(np.array(self.position) - np.array(self.goal_world_pos))
        if distance_to_goal < 0.2:
            return 0  # Stop action


        self.planned_action = action  # 保存规划的动作

        # if action==0 and self.zaizouyibu == 0:
        #     action = 1
        #     self.zaizouyibu = 1
        #     print('zaizouyibu')
        # elif self.zaizouyibu == 1:
        #     action = 0

        #     print('stop')
        return action

    
    def _world_to_global_map_coords(self, goal_world_pos, agent_world_pos, agent_rotation):
        """将世界坐标转换为全局地图坐标（固定的，不随agent移动变化）"""
        args = self.args
        
        # 使用与主程序相同的坐标转换逻辑
        goal_sim_x = -goal_world_pos[2]
        goal_sim_y = -goal_world_pos[0]
        
        agent_sim_x = -agent_world_pos[2] 
        agent_sim_y = -agent_world_pos[0]
        
        # 计算目标相对于agent的偏移（使用sim坐标系）
        offset_x = goal_sim_x - agent_sim_x
        offset_y = goal_sim_y - agent_sim_y
        
        # agent在全局地图中的初始位置（地图中心）
        agent_global_map_x = args.map_size_cm / 100.0 / 2.0
        agent_global_map_y = args.map_size_cm / 100.0 / 2.0
        
        # 计算目标在全局地图中的位置（米）
        goal_global_map_x = agent_global_map_x - offset_x
        goal_global_map_y = agent_global_map_y + offset_y
        
        # print(f"Agent初始全局地图位置: ({agent_global_map_x}, {agent_global_map_y})")
        # print(f"Sim坐标偏移: ({offset_x:.3f}, {offset_y:.3f})")
        # print(f"目标全局地图位置: ({goal_global_map_x:.3f}, {goal_global_map_y:.3f})")
        
        return goal_global_map_x, goal_global_map_y
    



    def _update_goal_map(self, map_shape, pose_pred):
        """更新地图中的goal位置"""
        if self.goal_global_map_pos is None:
            return np.zeros(map_shape)
        
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = pose_pred
        args = self.args
        
        # goal_global_map_pos是目标在全局地图中的固定位置（米）
        goal_global_x, goal_global_y = self.goal_global_map_pos
        
        # 转换为全局地图像素坐标
        goal_pixel_r = int(goal_global_x * 100.0 / args.map_resolution)
        goal_pixel_c = int(goal_global_y * 100.0 / args.map_resolution)
        
        # 转换为当前局部地图坐标
        goal_local_x = goal_pixel_r - gx1
        goal_local_y = goal_pixel_c - gy1
        
        # print(f"=== 目标位置计算 ===")
        # print(f"Agent world pos: ({self.position}, {self.goal_world_pos})")
        # print(f"目标全局地图位置(米): ({goal_global_x:.3f}, {goal_global_y:.3f})")
        # print(f"目标全局像素坐标: ({goal_pixel_r}, {goal_pixel_c})")
        # print(f"当前规划窗口: gx1={gx1}, gx2={gx2}, gy1={gy1}, gy2={gy2}")
        # print(f"目标局部坐标: ({goal_local_x}, {goal_local_y})")
        # print(f"局部地图大小: {map_shape}")
        
        # 创建goal地图
        goal_map = np.zeros(map_shape)
        
        # 检查目标是否在当前局部地图范围内
        if 0 <= goal_local_x < map_shape[0] and 0 <= goal_local_y < map_shape[1]:
            # 在目标位置设置goal
            goal_radius = 0  # 可调整半径
            for dx in range(-goal_radius, goal_radius + 1):
                for dy in range(-goal_radius, goal_radius + 1):
                    new_x = goal_local_x + dx
                    new_y = goal_local_y + dy
                    if 0 <= new_x < map_shape[0] and 0 <= new_y < map_shape[1]:
                        goal_map[int(new_x), int(new_y)] = 1
            
            # print(f"✓ 目标在局部地图内，设置在 ({goal_local_x}, {goal_local_y})")
        else:
            # print(f"✗ 目标不在当前局部地图内")
            
            # 计算agent在局部地图中的位置
            agent_pixel_r = int(start_x * 100.0 / args.map_resolution)
            agent_pixel_c = int(start_y * 100.0 / args.map_resolution)
            agent_local_x = agent_pixel_r - gx1
            agent_local_y = agent_pixel_c - gy1
            
            # 计算指向目标的方向
            dir_x = goal_local_x - agent_local_x
            dir_y = goal_local_y - agent_local_y
            
            if abs(dir_x) > 0 or abs(dir_y) > 0:
                # 在地图边缘设置中间目标
                length = np.sqrt(dir_x**2 + dir_y**2)
                if length > 0:
                    norm_dir_x = dir_x / length
                    norm_dir_y = dir_y / length
                    
                    # 设置距离agent一定距离的中间目标
                    intermediate_distance = min(map_shape[0]//3, map_shape[1]//3)
                    intermediate_x = int(agent_local_x + norm_dir_x * intermediate_distance)
                    intermediate_y = int(agent_local_y + norm_dir_y * intermediate_distance)
                    
                    # 确保在地图边界内
                    intermediate_x = max(3, min(intermediate_x, map_shape[0] - 4))
                    intermediate_y = max(3, min(intermediate_y, map_shape[1] - 4))
                    
                    # 设置中间目标
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            new_x = intermediate_x + dx
                            new_y = intermediate_y + dy
                            if 0 <= new_x < map_shape[0] and 0 <= new_y < map_shape[1]:
                                goal_map[int(new_x), int(new_y)] = 1
                    
                    print(f"设置中间目标在 ({intermediate_x}, {intermediate_y})")
        
        return goal_map


    


    def _get_stg(self, grid, start, goal, planning_window):
        """Get short-term goal"""
        [gx1, gx2, gy1, gy2] = planning_window

        x1, y1, = 0, 0
        x2, y2 = grid.shape

        def add_boundary(mat, value=1):
            h, w = mat.shape
            new_mat = np.zeros((h + 2, w + 2)) + value
            new_mat[1:h + 1, 1:w + 1] = mat
            return new_mat

        traversible = skimage.morphology.binary_dilation(
            grid[x1:x2, y1:y2], self.selem) != True
        traversible[self.collision_map[gx1:gx2, gy1:gy2][x1:x2, y1:y2] == 1] = 0
        traversible[cv2.dilate(self.visited_vis[gx1:gx2, gy1:gy2][x1:x2, y1:y2], 
                            self.kernel) == 1] = 1

        traversible[int(start[0] - x1) - 1:int(start[0] - x1) + 2,
                    int(start[1] - y1) - 1:int(start[1] - y1) + 2] = 1

        traversible = add_boundary(traversible)
        goal = add_boundary(goal, value=0)

        planner = FMMPlanner(traversible)
        selem = skimage.morphology.disk(2)
        goal = skimage.morphology.binary_dilation(goal, selem) != True
        goal = 1 - goal * 1.
        planner.set_multi_goal(goal)

        state = [start[0] - x1 + 1, start[1] - y1 + 1]
        stg_x, stg_y, replan, stop = planner.get_short_term_goal(state)

        stg_x, stg_y = stg_x + x1 - 1, stg_y + y1 - 1
        return (stg_x, stg_y), replan, stop


    def _preprocess_obs(self, obs, use_seg=True):
        args = self.args
        # print("obs: ", obs)
        # print(obs.shape)
        # exit()
        obs = obs.transpose(1, 2, 0)
        rgb = obs[:, :, :3]
        self.rgb = rgb

        depth = obs[:, :, 3:4]
        # print(111111111)
        # print(np.min(depth),np.max(depth))
        
        semantic = obs[:,:,4:5].squeeze()
        # print("obs: ", semantic.shape)
        if args.use_gtsem or 1:
            self.rgb_vis = rgb
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))
            for i in range(16):
                sem_seg_pred[:,:,i][semantic == i+1] = 1
        else: 
            red_semantic_pred, semantic_pred = self._get_sem_pred(
                rgb.astype(np.uint8), depth, use_seg=use_seg)
            
            sem_seg_pred = np.zeros((rgb.shape[0], rgb.shape[1], 15 + 1))   
            for i in range(0, 15):
                # print(mp_categories_mapping[i])
                sem_seg_pred[:,:,i][red_semantic_pred == mp_categories_mapping[i]] = 1

            sem_seg_pred[:,:,0][semantic_pred[:,:,0] == 0] = 0
            sem_seg_pred[:,:,1][semantic_pred[:,:,1] == 0] = 0
            sem_seg_pred[:,:,3][semantic_pred[:,:,3] == 0] = 0
            sem_seg_pred[:,:,4][semantic_pred[:,:,4] == 1] = 1
            sem_seg_pred[:,:,5][semantic_pred[:,:,5] == 1] = 1

        # sem_seg_pred = self._get_sem_pred(
        #     rgb.astype(np.uint8), depth, use_seg=use_seg)

        depth = self._preprocess_depth(depth, args.min_depth, args.max_depth)
        # print(np.min(depth),np.max(depth))
        # exit()
        self.depth = depth
        ds = args.env_frame_width // args.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]
            sem_seg_pred = sem_seg_pred[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth, sem_seg_pred),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        # print(depth.shape)
        # exit()
        depth = depth[:, :, 0] * 1

        for i in range(depth.shape[1]):
            depth[:, i][depth[:, i] == 0.] = depth[:, i].max()

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        # print(np.min(depth),np.max(depth))
        # depth = min_d * 100.0 + depth * max_d * 100.0
        depth = min_d * 100.0 + depth * (max_d-min_d) * 100.0
        # depth = depth*1000.

        return depth

    def _get_sem_pred(self, rgb, depth, use_seg=True):
        if use_seg:
            image = torch.from_numpy(rgb).to(self.device).unsqueeze_(0).float()
            depth = torch.from_numpy(depth).to(self.device).unsqueeze_(0).float()
            with torch.no_grad():
                red_semantic_pred = self.red_sem_pred(image, depth)
                red_semantic_pred = red_semantic_pred.squeeze().cpu().detach().numpy()
            semantic_pred, self.rgb_vis = self.sem_pred.get_prediction(rgb)
            semantic_pred = semantic_pred.astype(np.float32)
        else:
            semantic_pred = np.zeros((rgb.shape[0], rgb.shape[1], 16))
            self.rgb_vis = rgb[:, :, ::-1]
        return red_semantic_pred, semantic_pred

    def _visualize(self, inputs):
        args = self.args
        dump_dir = "{}/dump/{}/".format(args.dump_location,
                                        args.exp_name)
        ep_dir = '{}/episodes/thread_{}/eps_{}/'.format(
            dump_dir, self.rank, self.episode_no)
        if not os.path.exists(ep_dir):
            os.makedirs(ep_dir)

        local_w = inputs['map_pred'].shape[0]

        map_pred = inputs['map_pred']
        exp_pred = inputs['exp_pred']
        map_edge = inputs['map_edge']
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = inputs['pose_pred']

        sem_map = inputs['sem_map_pred']

        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)

        sem_map += 5

        no_cat_mask = sem_map == 20
        map_mask = np.rint(map_pred) == 1
        exp_mask = np.rint(exp_pred) == 1
        vis_mask = self.visited_vis[gx1:gx2, gy1:gy2] == 1
        edge_mask = map_edge == 1

        sem_map[no_cat_mask] = 0
        m1 = np.logical_and(no_cat_mask, exp_mask)
        sem_map[m1] = 2

        m2 = np.logical_and(no_cat_mask, map_mask)
        sem_map[m2] = 1

        sem_map[vis_mask] = 3
        sem_map[edge_mask] = 3


        # 添加人类障碍物可视化 - 新增部分
        planning_window = [gx1, gx2, gy1, gy2]
        self.visualize_human_obstacles(sem_map, planning_window)

        # 可视化预测障碍物（新增）
        self.visualize_predicted_obstacles(sem_map, planning_window)


        selem = skimage.morphology.disk(4)
        goal_mat = 1 - skimage.morphology.binary_dilation(
            self.goal, selem) != True

        goal_mask = goal_mat == 1
        sem_map[goal_mask] = 4
        # print(sem_map.shape,np.max(sem_map),np.min(sem_map),sem_map.dtype)
        # exit()
        self.sem_map = sem_map


        color_pal = [int(x * 255.) for x in color_palette]

        # 添加新颜色
        # 在现有的color_pal.extend部分，添加融合预测的颜色
        color_pal.extend([
            255, 0, 0,    # 红色 - 新的人类障碍物
            255, 128, 0,  # 橙色 - 衰减中的人类障碍物
            255, 255, 0,  # 黄色 - 即将消失的人类障碍物
            255, 0, 255,  # 紫色 - 当前人类位置
            0, 0, 255,    # 深蓝色 - 高置信度直线预测
            64, 164, 255, # 浅蓝色 - 中等置信度直线预测
            0, 255, 255,  # 青色 - 低置信度直线预测
            0, 128, 0,    # 深绿色 - 高置信度融合预测  (新增)
            128, 255, 128,# 浅绿色 - 中等置信度融合预测 (新增)
            200, 255, 200,# 淡绿色 - 低置信度融合预测  (新增)
        ])

        sem_map_vis = Image.new("P", (sem_map.shape[1],
                                      sem_map.shape[0]))
        sem_map_vis.putpalette(color_pal)
        sem_map_vis.putdata(sem_map.flatten().astype(np.uint8))
        sem_map_vis = sem_map_vis.convert("RGB")
        sem_map_vis = np.flipud(sem_map_vis)

        sem_map_vis = sem_map_vis[:, :, [2, 1, 0]]
        sem_map_vis = cv2.resize(sem_map_vis, (480, 480),
                                 interpolation=cv2.INTER_NEAREST)
        self.vis_image[50:530, 15:655] = self.rgb_vis
        self.vis_image[50:530, 670:1150] = sem_map_vis
        
        pos = (
            (start_x * 100. / args.map_resolution - gy1)
            * 480 / map_pred.shape[0],
            (map_pred.shape[1] - start_y * 100. / args.map_resolution + gx1)
            * 480 / map_pred.shape[1],
            np.deg2rad(-start_o)
        )

        agent_arrow = vu.get_contour_points(pos, origin=(670, 50), size=10)
        color = (int(color_palette[11] * 255),
                 int(color_palette[10] * 255),
                 int(color_palette[9] * 255))
        cv2.drawContours(self.vis_image, [agent_arrow], 0, color, -1)

        if args.visualize:
            # Displaying the image
            cv2.imshow("Thread {}".format(self.rank), self.vis_image)
            cv2.waitKey(1)

        if args.print_images:
            fn = '{}/episodes/thread_{}/eps_{}/{}-{}-Vis-{}.png'.format(
                dump_dir, self.rank, self.episode_no,
                self.rank, self.episode_no, self.timestep)
            
            # 保存图片
            # if args.collect:
            #     fn = '{}/episodes/thread_{}/eps_{}/{}-rgb.png'.format(
            #         dump_dir, self.rank, self.episode_no,self.timestep)
            #     cv2.imwrite(fn,self.rgb_vis)
            #     fn = '{}/episodes/thread_{}/eps_{}/{}-depth.png'.format(
            #         dump_dir, self.rank, self.episode_no,self.timestep)
            #     cv2.imwrite(fn,self.depth)
            #     fn = '{}/episodes/thread_{}/eps_{}/{}-sem_map.png'.format(
            #         dump_dir, self.rank, self.episode_no,self.timestep)
            #     cv2.imwrite(fn,self.vis_image[50:530, 670:1150])

            
            # print(fn)
            # cv2.imwrite(fn, self.vis_image)

    # 1. 首先需要添加获取人类朝向的方法
    def get_same_height_agents_with_orientation(self):
        """获取与agent_0同一高度的其他agents（人类），包含朝向信息"""
        try:
            agent_0_state = self.env.sim.get_agent_state(0)
            agent_0_height = agent_0_state.position[1]
            
            same_height_agents = {}
            for agent_id in range(1, 7):
                try:
                    agent_state = self.env.sim.get_agent_state(agent_id)
                    agent_pos = agent_state.position
                    agent_rot = agent_state.rotation
                    
                    # 检查是否在有效位置
                    if agent_pos[0] < -90 or agent_pos[1] < -90 or agent_pos[2] < -90:
                        continue
                        
                    # 检查是否在同一高度
                    height_diff = abs(agent_pos[1] - agent_0_height)
                    if height_diff <= self.same_height_threshold:
                        # 使用和主函数相同的方法计算朝向
                        x = -agent_pos[2]
                        y = -agent_pos[0]
                        
                        # 计算朝向角度
                        import quaternion
                        axis = quaternion.as_euler_angles(agent_rot)[0]
                        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
                            o = quaternion.as_euler_angles(agent_rot)[1]
                        else:
                            o = 2 * np.pi - quaternion.as_euler_angles(agent_rot)[1]
                        if o > np.pi:
                            o -= 2 * np.pi
                        
                        same_height_agents[agent_id] = {
                            'position': [agent_pos[0], agent_pos[2]],
                            'full_position': agent_pos,
                            'orientation': o,  # 朝向角度（弧度）
                            'sim_position': [x, y]  # sim坐标系位置
                        }
                except:
                    continue
            
            return same_height_agents
        except:
            return {}
        


   
    # 3. 修改update_human_obstacle_memory方法，添加朝向信息记录
    def update_human_obstacle_memory(self, planning_window):
        """更新人类障碍物记忆 - 包含朝向信息"""
        current_timestep = self.timestep
        args = self.args
        
        # 获取当前人类位置和朝向
        same_height_agents = self.get_same_height_agents_with_orientation()
        
        for agent_id, agent_info in same_height_agents.items():
            world_x, world_z = agent_info['position']
            orientation = agent_info['orientation']
            
            # 构造3D坐标
            human_world_pos = [world_x, 0, world_z]
            
            # 使用和goal相同的坐标转换逻辑
            human_global_map_x, human_global_map_y = self._world_to_global_map_coords_for_humans(
                human_world_pos
            )
            
            # 存储人类的中心点位置和朝向
            key = f"{human_global_map_x:.4f}_{human_global_map_y:.4f}"
            
            # 更新时间戳和朝向信息
            self.human_obstacle_memory[key] = {
                'timestamp': current_timestep,
                'orientation': orientation,
                'agent_id': agent_id
            }
            
            # 记录这个中心点曾经被人类影响过
            self.human_affected_areas.add(key)
        
        # print(f"人类中心点记忆数量: {len(self.human_obstacle_memory)}")
        
        # print(f"人类中心点记忆数量: {len(self.human_obstacle_memory)}")
        # print(f"人类影响过的中心点: {len(self.human_affected_areas)}")
    


    # 4. 相应地修改apply_human_obstacles_to_map方法
    def apply_human_obstacles_to_map(self, map_pred, planning_window):
        """使用形态学操作应用人类障碍物"""
        gx1, gx2, gy1, gy2 = planning_window
        current_timestep = self.timestep
        args = self.args
        
        # 复制原始地图
        modified_map = map_pred.copy()
        
        # 创建人类障碍物地图
        human_active_map = np.zeros_like(map_pred)
        human_expired_map = np.zeros_like(map_pred)
        
        # 处理所有人类中心点
        for key in self.human_affected_areas:
            global_x_meters, global_y_meters = map(float, key.split('_'))
            
            # 转换为局部地图坐标
            global_pixel_x = int(global_x_meters * 100.0 / args.map_resolution)
            global_pixel_y = int(global_y_meters * 100.0 / args.map_resolution)
            
            local_x = global_pixel_x - gx1
            local_y = global_pixel_y - gy1
            
            # 检查是否在当前地图范围内
            if (0 <= local_x < map_pred.shape[0] and 
                0 <= local_y < map_pred.shape[1]):
                
                if key in self.human_obstacle_memory:
                    obstacle_info = self.human_obstacle_memory[key]
                    timestamp = obstacle_info['timestamp'] if isinstance(obstacle_info, dict) else obstacle_info
                    time_diff = current_timestep - timestamp
                    
                    if time_diff <= self.obstacle_decay_time:
                        # 还在有效期内，在对应地图上标记中心点
                        human_active_map[local_x, local_y] = 1
                    else:
                        # 过期了，在过期地图上标记中心点
                        human_expired_map[local_x, local_y] = 1
        
        # 使用形态学操作扩展人类影响区域
        radius_pixels = int(self.obstacle_radius * 100.0 / args.map_resolution)
        human_selem = skimage.morphology.disk(radius_pixels)
        
        # 扩展激活的人类障碍物区域
        if np.any(human_active_map):
            human_active_expanded = skimage.morphology.binary_dilation(
                human_active_map, human_selem
            )
            modified_map[human_active_expanded] = 1
        
        # 扩展过期的人类障碍物区域并清除
        if np.any(human_expired_map):
            human_expired_expanded = skimage.morphology.binary_dilation(
                human_expired_map, human_selem
            )
            modified_map[human_expired_expanded] = 0
        
        return modified_map

    


    # 5. 修改可视化方法
    def visualize_human_obstacles(self, sem_map, planning_window):
        """在语义地图上可视化人类障碍物 - 包含朝向信息"""
        gx1, gx2, gy1, gy2 = planning_window
        current_timestep = self.timestep
        args = self.args
        
        # 创建人类障碍物地图
        human_obstacle_map = np.zeros_like(sem_map)
        human_current_map = np.zeros_like(sem_map)
        
        # 标记人类障碍物记忆的中心点
        for key, obstacle_info in self.human_obstacle_memory.items():
            global_x_meters, global_y_meters = map(float, key.split('_'))
            
            # 转换为局部地图坐标
            global_pixel_x = int(global_x_meters * 100.0 / args.map_resolution)
            global_pixel_y = int(global_y_meters * 100.0 / args.map_resolution)
            
            local_x = global_pixel_x - gx1
            local_y = global_pixel_y - gy1
            
            if (0 <= local_x < sem_map.shape[0] and 
                0 <= local_y < sem_map.shape[1]):
                
                timestamp = obstacle_info['timestamp'] if isinstance(obstacle_info, dict) else obstacle_info
                time_diff = current_timestep - timestamp
                
                if time_diff <= self.obstacle_decay_time:
                    human_obstacle_map[local_x, local_y] = 1
        
        # 标记当前人类位置
        same_height_agents = self.get_same_height_agents_with_orientation()
        for agent_id, agent_info in same_height_agents.items():
            world_x, world_z = agent_info['position']
            
            # 构造3D坐标
            human_world_pos = [world_x, 0, world_z]
            
            # 使用和goal相同的坐标转换
            human_global_map_x, human_global_map_y = self._world_to_global_map_coords_for_humans(
                human_world_pos
            )
            
            human_pixel_x = int(human_global_map_x * 100.0 / args.map_resolution)
            human_pixel_y = int(human_global_map_y * 100.0 / args.map_resolution)
            
            local_x = human_pixel_x - gx1
            local_y = human_pixel_y - gy1
            
            if (0 <= local_x < sem_map.shape[0] and 
                0 <= local_y < sem_map.shape[1]):
                human_current_map[local_x, local_y] = 1
        
        # 使用形态学膨胀
        if np.any(human_obstacle_map):
            obstacle_radius = int(self.obstacle_radius * 100.0 / args.map_resolution)
            obstacle_selem = skimage.morphology.disk(obstacle_radius)
            
            human_obstacle_expanded = skimage.morphology.binary_dilation(
                human_obstacle_map, obstacle_selem
            )
            sem_map[human_obstacle_expanded] = 20  # 红色 - 人类障碍物记忆
        
        # 当前人类位置
        if np.any(human_current_map):
            current_radius = int(self.obstacle_radius * 100.0 / args.map_resolution)
            current_selem = skimage.morphology.disk(current_radius)
            
            human_current_expanded = skimage.morphology.binary_dilation(
                human_current_map, current_selem
            )
            sem_map[human_current_expanded] = 23  # 紫色 - 当前人类位置
    
    def apply_predicted_obstacles_to_map(self, map_pred, planning_window):
        """将预测障碍物应用到地图上"""
        gx1, gx2, gy1, gy2 = planning_window
        current_timestep = self.timestep
        args = self.args
        
        # 复制地图
        modified_map = map_pred.copy()
        
        # 创建预测障碍物地图
        predicted_obstacle_map = np.zeros_like(map_pred, dtype=np.float32)
        
        for key, obstacle_info in self.predicted_obstacles.items():
            position = obstacle_info['position']
            weight = obstacle_info['weight']
            timestamp = obstacle_info['timestamp']
            
            global_x_meters, global_y_meters = position
            
            # 转换为局部地图坐标
            global_pixel_x = int(global_x_meters * 100.0 / args.map_resolution)
            global_pixel_y = int(global_y_meters * 100.0 / args.map_resolution)
            
            local_x = global_pixel_x - gx1
            local_y = global_pixel_y - gy1
            
            # 检查是否在当前地图范围内
            if (0 <= local_x < map_pred.shape[0] and 
                0 <= local_y < map_pred.shape[1]):
                
                # 计算时间衰减
                time_diff = current_timestep - timestamp
                if time_diff <= self.predicted_obstacle_decay_time:
                    time_decay = 1.0 - (time_diff / self.predicted_obstacle_decay_time)
                    final_weight = weight * time_decay
                    
                    # 累积权重（如果多个预测点重叠）
                    predicted_obstacle_map[local_x, local_y] = max(
                        predicted_obstacle_map[local_x, local_y], 
                        final_weight
                    )
        
        # 使用形态学操作扩展预测障碍物区域
        if np.any(predicted_obstacle_map > 0.1):  # 只处理权重大于0.1的区域
            # 创建较小的结构元素用于预测障碍物
            pred_radius = max(int(self.obstacle_radius * 0.7 * 100.0 / args.map_resolution), 2)
            pred_selem = skimage.morphology.disk(pred_radius)
            
            # 对于权重较高的区域（> 0.5），设置为完全障碍物
            high_weight_mask = predicted_obstacle_map > 0.5
            if np.any(high_weight_mask):
                high_weight_expanded = skimage.morphology.binary_dilation(
                    high_weight_mask, pred_selem
                )
                modified_map[high_weight_expanded] = 1
                # print(f"添加了 {np.sum(high_weight_expanded)} 个高权重预测障碍物像素")
            
            # 对于中等权重的区域（0.2-0.5），部分影响通行性
            med_weight_mask = (predicted_obstacle_map > 0.2) & (predicted_obstacle_map <= 0.5)
            if np.any(med_weight_mask):
                med_weight_expanded = skimage.morphology.binary_dilation(
                    med_weight_mask, skimage.morphology.disk(pred_radius // 2)
                )
                # 只在原本可通行的区域设置障碍物
                modified_map[med_weight_expanded & (modified_map == 0)] = 1
                # print(f"添加了 {np.sum(med_weight_expanded & (modified_map == 0))} 个中权重预测障碍物像素")
        
        return modified_map

    # 6. 修改预测障碍物可视化，区分直线预测
    def visualize_predicted_obstacles(self, sem_map, planning_window):
        """可视化预测的障碍物 - 添加融合预测类型"""
        gx1, gx2, gy1, gy2 = planning_window
        current_timestep = self.timestep
        args = self.args
        
        # 创建不同类型的预测障碍物地图
        linear_high_map = np.zeros_like(sem_map)
        linear_med_map = np.zeros_like(sem_map)
        linear_low_map = np.zeros_like(sem_map)
        
        # 添加融合预测地图
        fused_high_map = np.zeros_like(sem_map)
        fused_med_map = np.zeros_like(sem_map)
        fused_low_map = np.zeros_like(sem_map)

        for key, obstacle_info in self.predicted_obstacles.items():
            position = obstacle_info['position']
            weight = obstacle_info['weight']
            timestamp = obstacle_info['timestamp']
            prediction_type = obstacle_info.get('prediction_type', 'unknown')
            
            global_x_meters, global_y_meters = position
            
            # 转换为局部地图坐标
            global_pixel_x = int(global_x_meters * 100.0 / args.map_resolution)
            global_pixel_y = int(global_y_meters * 100.0 / args.map_resolution)
            
            local_x = global_pixel_x - gx1
            local_y = global_pixel_y - gy1
            
            if (0 <= local_x < sem_map.shape[0] and 
                0 <= local_y < sem_map.shape[1]):
                
                # 计算综合权重
                time_diff = current_timestep - timestamp
                if time_diff <= self.predicted_obstacle_decay_time:
                    time_decay = 1.0 - (time_diff / self.predicted_obstacle_decay_time)
                    final_weight = weight * time_decay
                    
                    # 根据预测类型分类
                    if prediction_type == 'linear':
                        if final_weight > 0.7:
                            linear_high_map[local_x, local_y] = 1
                        elif final_weight > 0.4:
                            linear_med_map[local_x, local_y] = 1
                        elif final_weight > 0.2:
                            linear_low_map[local_x, local_y] = 1
                    elif prediction_type == 'fused':  # 新增的融合预测类型
                        if final_weight > 0.7:
                            fused_high_map[local_x, local_y] = 1
                        elif final_weight > 0.4:
                            fused_med_map[local_x, local_y] = 1
                        elif final_weight > 0.2:
                            fused_low_map[local_x, local_y] = 1

        # 初始化expanded变量
        linear_high_expanded = np.zeros_like(sem_map, dtype=bool)
        linear_med_expanded = np.zeros_like(sem_map, dtype=bool)
        linear_low_expanded = np.zeros_like(sem_map, dtype=bool)
        fused_high_expanded = np.zeros_like(sem_map, dtype=bool)
        fused_med_expanded = np.zeros_like(sem_map, dtype=bool)
        fused_low_expanded = np.zeros_like(sem_map, dtype=bool)

        # 显示直线预测（保持原有逻辑）
        if np.any(linear_high_map):
            linear_selem = skimage.morphology.rectangle(3, 6)
            linear_high_expanded = skimage.morphology.binary_dilation(linear_high_map, linear_selem)
            sem_map[linear_high_expanded] = 24  # 深蓝色
        
        if np.any(linear_med_map):
            linear_selem = skimage.morphology.rectangle(2, 4)
            linear_med_expanded = skimage.morphology.binary_dilation(linear_med_map, linear_selem)
            sem_map[linear_med_expanded & ~linear_high_expanded] = 25  # 浅蓝色
        
        if np.any(linear_low_map):
            linear_selem = skimage.morphology.rectangle(1, 3)
            linear_low_expanded = skimage.morphology.binary_dilation(linear_low_map, linear_selem)
            sem_map[linear_low_expanded & ~linear_high_expanded & ~linear_med_expanded] = 26  # 青色

        # 显示融合预测（使用不同的形状和颜色）
        if np.any(fused_high_map):
            fused_selem = skimage.morphology.disk(3)  # 使用圆形表示融合预测
            fused_high_expanded = skimage.morphology.binary_dilation(fused_high_map, fused_selem)
            sem_map[fused_high_expanded & ~linear_high_expanded] = 24  # 新颜色 - 深绿色
        
        if np.any(fused_med_map):
            fused_selem = skimage.morphology.disk(2)
            fused_med_expanded = skimage.morphology.binary_dilation(fused_med_map, fused_selem)
            sem_map[fused_med_expanded & ~fused_high_expanded & ~linear_high_expanded & ~linear_med_expanded] = 25  # 浅绿色
        
        if np.any(fused_low_map):
            fused_selem = skimage.morphology.disk(1)
            fused_low_expanded = skimage.morphology.binary_dilation(fused_low_map, fused_selem)
            all_high_med = (fused_high_expanded | fused_med_expanded | 
                        linear_high_expanded | linear_med_expanded | linear_low_expanded)
            sem_map[fused_low_expanded & ~all_high_med] = 26   # 淡绿色


    def _world_to_global_map_coords_for_humans(self, human_world_pos):
        """将人类世界坐标转换为全局地图坐标（使用和goal相同的逻辑）"""
        args = self.args
        
        # 添加调试：确认使用的是固定参考系
        # print(f"使用的参考位置: {self.initial_agent_position}")
        # print(f"当前agent位置: {self.position}")


        # 使用初始agent位置作为固定参考系，而不是当前位置
        agent_world_pos = self.initial_agent_position
        

        # 使用与goal相同的坐标转换逻辑
        human_sim_x = -human_world_pos[2]
        human_sim_y = -human_world_pos[0]
        
        agent_sim_x = -agent_world_pos[2] 
        agent_sim_y = -agent_world_pos[0]
        
        # 计算人类相对于初始agent的偏移（使用sim坐标系）
        offset_x = human_sim_x - agent_sim_x
        offset_y = human_sim_y - agent_sim_y
        
        # agent在全局地图中的初始位置（地图中心）
        agent_global_map_x = args.map_size_cm / 100.0 / 2.0
        agent_global_map_y = args.map_size_cm / 100.0 / 2.0
        
        # 计算人类在全局地图中的位置（米）
        human_global_map_x = agent_global_map_x - offset_x
        human_global_map_y = agent_global_map_y + offset_y
        # print(f"Agent位置: {agent_world_pos}")
        # print(f"人类位置: {human_world_pos}")
        # print(f"Offset: ({offset_x:.4f}, {offset_y:.4f})")
        # print(f"转换后: ({human_global_map_x:.4f}, {human_global_map_y:.4f})")

        return human_global_map_x, human_global_map_y
    

    def predict_orientation_based_trajectory(self, agent_info):
        """基于朝向的轨迹预测 - 从现有代码提取"""
        world_x, world_z = agent_info['position']
        orientation = agent_info['orientation']
        
        # 构造3D坐标
        human_world_pos = [world_x, 0, world_z]
        
        # 转换为全局地图坐标
        human_global_map_x, human_global_map_y = self._world_to_global_map_coords_for_humans(
            human_world_pos
        )
        
        # 修正朝向角度到地图坐标系
        map_orientation = orientation + np.pi/2
        direction_map_x = np.cos(map_orientation)
        direction_map_y = -np.sin(map_orientation)
        
        # 生成轨迹点
        trajectory_points = []
        for step in range(1, self.prediction_steps + 1):
            distance = (step / self.prediction_steps) * self.prediction_distance
            
            next_x = human_global_map_x + direction_map_x * distance
            next_y = human_global_map_y + direction_map_y * distance
            
            trajectory_points.append([next_x, next_y])
        
        return trajectory_points
    
    # def fused_trajectory_prediction(self, agent_id, agent_info):
    #     """融合历史轨迹和朝向信息的预测"""
        
    #     # 获取历史轨迹预测
    #     if hasattr(self, 'trajectory_predictor'):
    #         history_prediction = self.trajectory_predictor.predict_trajectory(agent_id)
    #     else:
    #         history_prediction = None
        

    #     # print(len(history_prediction))
    #     # 获取朝向预测
    #     orientation_prediction = self.predict_orientation_based_trajectory(agent_info)
        
    #     if history_prediction is not None and len(history_prediction) > 0:
    #         # 融合两种预测
    #         fused_trajectory = []
    #         history_weight = 0.1  # 历史轨迹权重
            
    #         # 确保两个预测长度一致
    #         min_length = min(len(history_prediction), len(orientation_prediction))
            
    #         for i in range(min_length):
    #             hist_pos = history_prediction[i]
    #             orient_pos = orientation_prediction[i]
                
    #             # 距离越远，越依赖朝向预测
    #             dynamic_hist_weight = history_weight * (0.8 ** i)
    #             dynamic_orient_weight = 1 - dynamic_hist_weight
                
    #             fused_pos = (
    #                 dynamic_hist_weight * np.array(hist_pos) +
    #                 dynamic_orient_weight * np.array(orient_pos)
    #             )
    #             fused_trajectory.append(fused_pos.tolist())
    #     else:
    #         # 只使用朝向预测
    #         fused_trajectory = orientation_prediction
        
    #     return fused_trajectory
    
    def fused_trajectory_prediction(self, agent_id, agent_info):
        """融合历史轨迹和朝向信息的预测"""
        
        # 朝向预测作为基础
        orientation_prediction = self.predict_orientation_based_trajectory(agent_info)
        
        # 历史预测仅用于前几步的微调
        if hasattr(self, 'trajectory_predictor'):
            history_prediction = self.trajectory_predictor.predict_trajectory(agent_id)
            
            if history_prediction and len(history_prediction) > 0:
                fused_trajectory = []
                adjustment_steps = min(3, len(orientation_prediction))  # 只调整前3步
                
                for i, orient_pos in enumerate(orientation_prediction):
                    if i < adjustment_steps and i < len(history_prediction):
                        # 只对前几步进行轻微调整
                        hist_pos = history_prediction[i]
                        adjustment_weight = 0.1 * (0.3 ** i)  # 快速衰减
                        
                        adjusted_pos = (
                            (1 - adjustment_weight) * np.array(orient_pos) +
                            adjustment_weight * np.array(hist_pos)
                        )
                        fused_trajectory.append(adjusted_pos.tolist())
                    else:
                        # 后续步骤完全使用朝向预测
                        fused_trajectory.append(orient_pos)
                
                return fused_trajectory
        
        # 默认返回朝向预测
        return orientation_prediction

    def update_trajectory_prediction(self, planning_window):
        """更新轨迹预测 - 使用融合方法"""
        current_timestep = self.timestep
        args = self.args
        
        # 清空之前的预测障碍物
        self.predicted_obstacles = {}
        
        # 获取当前所有人类位置和朝向
        same_height_agents = self.get_same_height_agents_with_orientation()
        
        for agent_id, agent_info in same_height_agents.items():

            world_x, world_z = agent_info['position']
        
            # 转换为全局地图坐标
            human_world_pos = [world_x, 0, world_z]
            human_global_map_x, human_global_map_y = self._world_to_global_map_coords_for_humans(human_world_pos)
            current_pos = [human_global_map_x, human_global_map_y]
            
            # **关键修复：更新轨迹预测器的历史数据**
            if hasattr(self, 'trajectory_predictor'):
                self.trajectory_predictor.update_agent_position(agent_id, current_pos, current_timestep)
                # print(f"[DEBUG] 更新 Agent {agent_id} 位置: {current_pos}")
                
                # 检查历史数据
                # trajectory = self.trajectory_predictor.agent_trajectories[agent_id]
                # print(f"[DEBUG] Agent {agent_id} 历史长度: {len(trajectory)}")
                # if len(trajectory) > 0:
                #     positions = [entry['position'] for entry in trajectory]
                #     print(f"[DEBUG] Agent {agent_id} 最近3个位置: {positions[-3:]}")
                    
                #     # 立即测试预测
                #     predicted = self.trajectory_predictor.predict_trajectory(agent_id)
                #     print(f"[DEBUG] Agent {agent_id} 预测轨迹长度: {len(predicted) if predicted else 0}")
            # 使用融合预测方法
            fused_trajectory = self.fused_trajectory_prediction(agent_id, agent_info)
            
            # 将融合轨迹转换为障碍物
            for step, future_pos in enumerate(fused_trajectory):
                obstacle_x, obstacle_y = future_pos
                
                # 在轨迹点周围创建障碍物区域
                width_steps = self.width_steps
                for width_step in range(-width_steps//2, width_steps//2 + 1):
                    width_offset = (width_step / width_steps) * self.prediction_width
                    
                    # 计算垂直于移动方向的偏移
                    if step > 0:
                        prev_pos = fused_trajectory[step-1] if step > 0 else [obstacle_x, obstacle_y]
                        direction = np.array([obstacle_x - prev_pos[0], obstacle_y - prev_pos[1]])
                        if np.linalg.norm(direction) > 0:
                            direction = direction / np.linalg.norm(direction)
                            perpendicular = np.array([-direction[1], direction[0]])
                        else:
                            perpendicular = np.array([0, 1])
                    else:
                        perpendicular = np.array([0, 1])
                    
                    final_x = obstacle_x + perpendicular[0] * width_offset
                    final_y = obstacle_y + perpendicular[1] * width_offset
                    
                    # 计算权重
                    distance_weight = 1.0 - (step / len(fused_trajectory)) * self.distance_decay_factor
                    width_weight = 1.0 - abs(width_offset) / (self.prediction_width / 2) * self.width_decay_factor
                    final_weight = distance_weight * width_weight
                    
                    if final_weight > self.min_prediction_weight:
                        key = f"{final_x:.2f}_{final_y:.2f}_fused_{agent_id}_{step}"
                        
                        self.predicted_obstacles[key] = {
                            'timestamp': current_timestep,
                            'weight': final_weight,
                            'position': [final_x, final_y],
                            'future_step': step,
                            'agent_id': agent_id,
                            'prediction_type': 'fused'
                        }
        
        # print(f"融合预测障碍物数量: {len(self.predicted_obstacles)}")







