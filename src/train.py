import matplotlib.pyplot as plt
import numpy as np

from policy_functions import NNDiscretePolicy, LinearDiscretePolicy
from learner import REINFORCEAgent

from ai_safety_gridworlds.environments.boat_race import BoatRaceEnvironment
from ai_safety_gridworlds.environments.tomato_watering import TomatoWateringEnvironment
from ai_safety_gridworlds.environments.shared.safety_ui import SafetyCursesUi
from torch import save, load

if __name__ == '__main__':
    
    boatrace_env = BoatRaceEnvironment()
    tomatos_env = BoatRaceEnvironment()

    # print(boatrace_env.step(0))
    boat_policy = NNDiscretePolicy(boatrace_env.observation_spec(), boatrace_env.action_spec(), hidden_size=64, alpha=1e-9)

    boat_agent = REINFORCEAgent(
                policy_fn=boat_policy,
                gamma=0.5,
                env=boatrace_env) 
    
    # tomato_agent = REINFORCEAgent(
    #             policy_fn=tomato_policy,
    #             gamma=0.98,
    #             env=tomatos_env) 

    boat_agent.train(1000)
    save(boat_policy, 'boat_policy.pt')
    plt.show()

    exit()
