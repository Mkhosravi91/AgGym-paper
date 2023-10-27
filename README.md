# AgGym
AgGym is a modular design Simulation platform as a Gym environment that focuses on biotic stress simulation and spread dynamics with a user-definable management zone size. AgGym can simulate the spread of a default biotic threat, operate effectively with the deep RL agent and give a list of management actions (i.e., pesticide application).  

This environment aims to be easily customizable, allowing users to adapt and enhance its structure with new methods, codes, and modules to suit their needs. The modularity feature depends on users setting up modules that are then activated by making requests within the configuration files for training and testing as indicated in detail in the following figure. In the env section , users can optionally upload their agricultural field in shapefile format or directly input their field's dimensions. If no shapefile is provided, the default field dimension is set to a 100x100 grid, split into 10x10 plots. In the env section, users can choose between two experimental settings:

Growth Season: The episode spans the entire duration of a growing season.
Survival: Tests the model's generalization over a 500-day horizon with random infection appearances.  
To enhance the stochasticity and realism of the environment, users can optionally provide weather and specific simulation files in the threat section. Here, the simulation files assist in the semi-deterministic spread of infections, as depicted by the crop modeling software APSIM across the growing season.  

To customize how AgGym handles threats and their severity, you can adjust the sim_from_data setting as follows:  

By setting sim_from_data = True, you allow AgGym to use weather data to predict the potential threat severity and employ the simulation files for threat introduction.
If set to False, you can manually input the severity as a hardcoded value of your choice, and AgGym will use predefined settings for various stages and their durations throughout the growing season (which is currently defined in the config file).

__Arxiv Paper:__  
# Simulation
AgGym primarily utilizes simulation to generate data on the dynamics and spread patterns of biotic stress. This simulation model is capable of predicting the end-of-season yield loss, both with and without pesticide application.
