import numpy as np
from collections import defaultdict, deque

class HumanTrajectoryPredictor:
    def __init__(self, history_length=5, prediction_steps=10, prediction_interval=3):
        """
        Args:
            history_length: 用于拟合的历史轨迹长度
            prediction_steps: 预测未来多少步
            prediction_interval: 每隔多少步进行一次预测
        """
        self.history_length = history_length
        self.prediction_steps = prediction_steps
        self.prediction_interval = prediction_interval
        
        # 存储每个agent的历史轨迹
        self.agent_trajectories = defaultdict(lambda: deque(maxlen=self.history_length))
        
        # 存储预测的轨迹点
        self.predicted_trajectories = {}
        
        # 记录上次预测的时间步
        self.last_prediction_step = -1
    

    def reset(self):
        """重置所有轨迹数据"""
        self.agent_trajectories.clear()
        self.predicted_trajectories.clear()
        self.last_prediction_step = -1


    def update_agent_position(self, agent_id, position, timestep):
        """更新agent的位置历史"""
        self.agent_trajectories[agent_id].append({
            'position': position,
            'timestep': timestep
        })

    def should_predict(self, current_timestep):
        """判断是否应该进行预测"""
        return (current_timestep - self.last_prediction_step) >= self.prediction_interval

    def predict_trajectory(self, agent_id):
        """为指定agent预测轨迹"""
        trajectory = self.agent_trajectories[agent_id]
        
        if len(trajectory) < 2:
            return []
        
        # 提取位置和时间
        positions = [entry['position'] for entry in trajectory]
        timesteps = [entry['timestep'] for entry in trajectory]
        
        # 如果只有一个点或者位置没有变化，返回静止预测
        # 修改为：
        if len(set([tuple(pos) for pos in positions])) == 1:
            return [positions[-1]] * self.prediction_steps
        
        # 线性拟合
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        # 使用numpy进行线性拟合
        if len(timesteps) >= 2:
            # 拟合x坐标
            x_poly = np.polyfit(timesteps, x_coords, 1)
            # 拟合y坐标  
            y_poly = np.polyfit(timesteps, y_coords, 1)
            
            # 预测未来位置
            predicted_positions = []
            last_timestep = timesteps[-1]
            
            for i in range(1, self.prediction_steps + 1):
                future_timestep = last_timestep + i
                future_x = np.polyval(x_poly, future_timestep)
                future_y = np.polyval(y_poly, future_timestep)
                predicted_positions.append([future_x, future_y])
            
            return predicted_positions
        
        return []

    def predict_all_trajectories(self, current_timestep):
        """预测所有agent的轨迹"""
        if not self.should_predict(current_timestep):
            return self.predicted_trajectories
        
        self.last_prediction_step = current_timestep
        self.predicted_trajectories = {}
        
        for agent_id in self.agent_trajectories:
            predicted = self.predict_trajectory(agent_id)
            if predicted:
                self.predicted_trajectories[agent_id] = predicted
        
        return self.predicted_trajectories

    def get_predicted_obstacles(self, current_timestep, decay_start_step=5):
        """获取预测轨迹上的障碍物点，带有时间衰减"""
        obstacles = []
        
        for agent_id, trajectory in self.predicted_trajectories.items():
            for step_idx, position in enumerate(trajectory):
                # 计算这个预测点对应的未来时间步
                future_step = step_idx + 1
                
                # 添加衰减权重，距离当前时间越远权重越小
                if future_step <= decay_start_step:
                    weight = 1.0
                else:
                    # 线性衰减
                    weight = max(0.1, 1.0 - (future_step - decay_start_step) * 0.1)
                
                obstacles.append({
                    'position': position,
                    'agent_id': agent_id,
                    'future_step': future_step,
                    'weight': weight,
                    'timestamp': current_timestep  # 记录预测时的时间戳
                })
        
        return obstacles