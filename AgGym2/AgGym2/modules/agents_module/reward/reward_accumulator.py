import importlib
import pdb
def init(self, severity=0):
	reward_module_list = self.reward.split(" + ")
	self.reward_module_dict = {}
	for i in reward_module_list:
		self.reward_module_dict[i] = importlib.import_module(f'modules.agents_module.reward.{i}')

def main(self):
	reward = 0
	for mod in self.reward_module_dict.values():
		reward += mod.main(self)
		
	return reward

