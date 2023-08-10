import numpy as np

def main(self, severity=0):
    pesticide_reward = {0: 0., 1: -.2, 2:-.5, 3:-.9}
    if pesticide_reward[self.action] != 0 and len(self.threat.infect_list) == 0:
        # reward = (self.state_space) * pesticide_reward[self.action] * self.unit_pesticide_per_acre
        reward = (self.state_space) * -(self.unit_pesticide_per_acre[self.action])
    else:
        reward = pesticide_reward[self.action] * len(self.threat.infect_list) * self.unit_pesticide_per_acre
        reward = len(self.threat.infect_list) * -(self.unit_pesticide_per_acre[self.action])
    print(f"pesticide reward: ({reward}) = ({pesticide_reward[self.action] * len(self.threat.infect_list)})")
    # print('yield pesticide={}'.format(reward))
    return reward

