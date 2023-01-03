from azureml.core import Workspace, Dataset, Environment
from azureml.core import Experiment, ScriptRunConfig

# define the experiment

ws = Workspace.from_config()
env = Environment.get(workspace=ws, name="PyTorch")

# Get the registered dataset from azure
dataset = Dataset.get_by_name(ws, name='crash_images')

# get our compute target
compute_target = ws.compute_targets["gpu-cluster-2"]
exp = Experiment(workspace=ws, name='test_training')

# setup the run details
src = ScriptRunConfig(source_directory="../models",
                      script='model_training.py',
                      compute_target=compute_target,
                      environment=env)

# Submit the model to azure!
run = exp.submit(config=src)
