import numpy as np
import pdb

# def main(self, severity=0):
#     reward = 0
#     for x, y in self.threat.infect_list:
#         if self.threat.action_value !=0:
#             reward += (self.threat.infect_list[x, y]) * -(self.unit_pesticide_per_acre[self.threat.action_matrix[x//10, y//10]])


reward_history = []  
def main(self, severity=0):
    
    # for x, y in self.threat.infect_list:

#     for i in self.action:
#         reward += -(self.unit_pesticide_per_acre[i])
        reward = 0
        for coord, value in self.threat.action_map.items():
                # pdb.set_trace()
                if coord not in self.threat.infect_list_before_pest and coord in self.threat.healthy and value != 0:
                        reward += -(self.unit_pesticide_per_acre[value])-0.1
                elif (coord not in self.threat.infect_list_before_pest) and (coord in self.threat.infect_sprayed) or \
                (coord in self.threat.alive_sprayed) and value != 0:
                        reward += -(self.unit_pesticide_per_acre[value])-0.2
                        # for i in self.action:
                elif coord not in self.threat.infect_list_before_pest and value != 0:
                        reward += -(self.unit_pesticide_per_acre[value])
    # for idx, k in enumerate(self.threat.infect_list):
    #     x, y = k
    #     if self.threat.action_value != 0:
            # Use separate integer indices for accessing elements
            # x_index = x // 10
            # y_index = y // 10
            # infect_value = self.threat.infect_list[x][y]
            # print((x_index, y_index))
            # reward += self.unit_pesticide_per_acre[int(self.threat.action_matrix[x_index, y_index])]
            # pesticide_value = self.unit_pesticide_per_acre[self.threat.action_matrix[x_index][y_index]]
            # reward += infect_value * (-pesticide_value)

    # pesticide_reward = {0: 0., 1: -.2, 2:-.5, 3:-.9}
    # if pesticide_reward[self.action] != 0 and len(self.threat.infect_list) == 0:
    #     reward = (self.state_space) * -(self.unit_pesticide_per_acre[self.action])
    # else:
    #     # reward = len(self.threat.infect_list_before_pest) * -(self.unit_pesticide_per_acre[self.action])
    #     reward = len(self.threat.infect_list) * -(self.unit_pesticide_per_acre[self.action])
    # print(len(self.threat.infect_list_before_pest))   
    # print(len(self.threat.infect_list))
    # print(f"pesticide reward: ({reward}) = ({pesticide_reward[self.action] * len(self.threat.infect_list)})")
    # print(f"pesticide reward: ({reward}) = ({-(self.unit_pesticide_per_acre[self.action]) * len(self.threat.infect_list)})")
    # print((self.state_space) * -(self.unit_pesticide_per_acre[self.action]))
    # print(np.round((self.state_space) * pesticide_reward[self.action] * (int(17/self.num_of_plots_per_acre )), 5))
    # print(np.round(len(self.threat.infect_list)* pesticide_reward[self.action] * (int(17/self.num_of_plots_per_acre )),5))
    # print(-(self.unit_pesticide_per_acre[self.action]))
    # # print('yield pesticide={}'.format(reward))
        print(f"pesticide reward: ({reward})")
        reward_history.append(reward)
        return reward



