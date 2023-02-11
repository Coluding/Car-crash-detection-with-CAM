from azureml.core.model import Model
from azureml.core import Workspace, Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, LocalWebservice

ws = Workspace(
    subscription_id="a0375d6b-a4f6-4c9e-9054-e9982f6f6765",
    resource_group="GPU_Test",
    workspace_name="InsuranceImageRecognition",
    _location="westeu"
)
model = Model(ws, "EffNet")

env = Environment.get(workspace=ws, name="insurance_image_recog")
inference_config = InferenceConfig(entry_script="../models/inference_run.py", environment=env)


aci_service_name = "test-deploy"

#deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
deployment_config = LocalWebservice.deploy_configuration(port=8890)

service = Model.deploy(ws, aci_service_name, [model], inference_config, deployment_config, overwrite=True)
service.wait_for_deployment(True)

print(service.state)