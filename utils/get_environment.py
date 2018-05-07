"""
    Add the environment you wish to train here
"""

from visual_im.utils.dm_env import DeepMindEnv

def get_environment(domain_name=None, task_name=None):
    if domain_name is None or task_name is None: print("Need to specify domain name and task name for the environment")
    return DeepMindEnv(domain_name, task_name)
