from azureml.core import Workspace, Dataset, Environment
from azureml.core import Experiment, ScriptRunConfig


# define the experiment

ws = Workspace(
    subscription_id="a0375d6b-a4f6-4c9e-9054-e9982f6f6765",
    resource_group="GPU_Test",
    workspace_name="InsuranceImageRecognition",
    _location="westeu"
)
env = Environment.get(workspace=ws, name="insurance_image_recog")

# get our compute target
compute_target = ws.compute_targets["bierli011"]
exp = Experiment(workspace=ws, name='test_training')

dataset = Dataset.get_by_name(ws, "crash_images")



# setup the run details
src = ScriptRunConfig(source_directory="../models",
                      script='model_training.py',
                      arguments=['--data-path', dataset.as_mount()],
                      compute_target=compute_target,
                      environment=env)

# Submit the model to azure!
run = exp.submit(config=src)
