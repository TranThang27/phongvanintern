import numpy as np

# Thông số giả lập
sound_speed = 343.0  # ví dụ truyền bằng sóng âm m/s
master_gps = np.array([12.22, 17.13])  # Tọa độ GPS master 
master_pos = np.array([0.0, 0.0, 0.0])  # vị trí master tương đối 
master_angle = np.deg2rad(30.0)  # Góc so với cực Bắc

time_slave1 = 0.002  # Thời gian truyền master->slave1
time_slave2 = 0.006  
compass_s1 = np.deg2rad(60.0)  # Góc slave1 so với cực bắc
compass_s2 = np.deg2rad(75.0)  

slave1_dis = sound_speed * time_slave1  # Khoảng cách master->slave1
slave2_dis = sound_speed * time_slave2  # Khoảng cách master->slave2

theta1 = compass_s1 - master_angle  # Góc giữa slave1 và master
theta2 = compass_s2 - master_angle  # Góc giữa slave2 và master

pos_s1 = np.array([slave1_dis * np.cos(theta1), slave1_dis * np.sin(theta1), 0.0]) # vị trí slave1 trên lưới tọa độ
pos_s2 = np.array([slave2_dis * np.cos(theta2), slave2_dis * np.sin(theta2), 0.0]) #vị trí slave2 trên lưới tọa độ

class kalmanfilter:
    def __init__(self):
        self.x = np.array([0.0, 0.0, 0.0, 0.0])  
        self.P = np.eye(4)  
        self.Q = np.eye(4) * 0.01  
        self.R = np.eye(2) * 0.1  
        self.F = np.array([[1, 0, 0.5, 0], [0, 1, 0, 0.5], [0, 0, 1, 0], [0, 0, 0, 1]])  
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  

    def update(self, z):
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(4) - K @ self.H) @ P_pred
        return self.x[:2]

def calculate_rotation_and_distance(theta_relative, distance, slave_name):
    theta_deg = np.rad2deg(theta_relative)
    if theta_deg >= 0:
        rotation = f"Quay trái {theta_deg:.2f} độ"
    else:
        rotation = f"Quay phải {abs(theta_deg):.2f} độ"
    distance_str = f"Di chuyển {distance:.3f} mét"
    return rotation, distance_str

kf_s1 = kalmanfilter()
kf_s2 = kalmanfilter()

#Thêm nhiễu thử bộ lọc
np.random.seed(42)
noise = 0.05  
z_s1 = pos_s1[:2] + np.random.normal(0, noise, 2)
z_s2 = pos_s2[:2] + np.random.normal(0, noise, 2)

pos_s1_filtered = kf_s1.update(z_s1)
pos_s2_filtered = kf_s2.update(z_s2)

rotation_s1, distance_s1 = calculate_rotation_and_distance(theta1, slave1_dis, "Slave 1")
rotation_s2, distance_s2 = calculate_rotation_and_distance(theta2, slave2_dis, "Slave 2")

print(f"Master - Tọa độ GPS: ({master_gps[0]:.2f}, {master_gps[1]:.2f}) độ")
print(f"Slave 1 - Tọa độ tương đối: ({pos_s1_filtered[0]:.3f}, {pos_s1_filtered[1]:.3f}) m")
print(f"Slave 1 - {rotation_s1}")
print(f"Slave 1 - {distance_s1}")
print(f"Slave 2 - Tọa độ tương đối: ({pos_s2_filtered[0]:.3f}, {pos_s2_filtered[1]:.3f}) m")
print(f"Slave 2 - {rotation_s2}")
print(f"Slave 2 - {distance_s2}")
