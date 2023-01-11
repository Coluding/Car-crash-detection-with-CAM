# Handle to the workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import AmlCompute, ComputeTarget


from azureml.core import Workspace

credential = DefaultAzureCredential()
"""ws = Workspace.get(name='InsuranceImageRecognition',
                   subscription_id='a0375d6b-a4f6-4c9e-9054-e9982f6f6765',
                   resource_group='GPU_Test'
                   )

list_vms = AmlCompute.supported_vmsizes(workspace=ws)

compute_config = RunConfiguration()
compute_config.target = "amlcompute"
compute_config.amlcompute.vm_size = "STANDARD_D1_V2"""

ws = Workspace(
    subscription_id="a0375d6b-a4f6-4c9e-9054-e9982f6f6765",
    resource_group="GPU_Test",
    workspace_name="InsuranceImageRecognition",
    _location="westeu"
)

ws.write_config(path='.azureml')
# the name we are going to use to reference our cluster
compute_name = "gpu-cluster-NC6"

# the azure machine type
vm_size = 'Standard_NC6_Promo'

# define the cluster and the max and min number of nodes
provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                            min_nodes = 0,
                                                            max_nodes = 10)
# create the cluster
#compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# create an enviornment
env = Environment(name='insurance_image_recog')
                                                 # "file:///D:\ML\DL\projects\insurance_image_recog\environment.yml")


env.environment_variables['CUDA_ALLOC_CONF'] = "max_split_size_mb:64"


"""# define packages for image
cd = CondaDependencies.create(pip_packages=[
    "absl-py==1.3.0",
    "alembic==1.8.1",
    "asgiref==3.5.2",
    "astunparse==1.6.3",
    "cachetools==5.2.0",
    "charset-normalizer==2.1.1",
    "click==8.1.3",
    "cloudpickle==2.2.0",
    "colorama==0.4.6",
    "databricks-cli==0.17.3",
    "django==4.1.3",
    "docker==6.0.1",
    "entrypoints==0.4",
    "app==2.2.2",
    "flatbuffers==22.10.26",
    "gast==0.4.0",
    "gitdb==4.0.9",
    "gitpython==3.1.29",
    "google-auth==2.14.1",
    "google-auth-oauthlib==0.4.6",
    "google-pasta==0.2.0",
    "greenlet==2.0.1",
    "grpcio==1.50.0",
    "h5py==3.7.0",
    "idna==3.4",
    "importlib-metadata==5.0.0",
    "itsdangerous==2.1.2",
    "jinja2==3.1.2",
    "keras==2.10.0",
    "keras-preprocessing==1.1.2",
    "libclang==14.0.6",
    "llvmlite==0.39.1",
    "mako==1.2.4",
    "markdown==3.4.1",
    "markupsafe==2.1.1",
    "mlflow==2.0.1",
    "numba==0.56.4",
    "oauthlib==3.2.2",
    "opt-einsum==3.3.0",
    "protobuf==3.19.6",
    "pyarrow==10.0.0",
    "pyasn1==0.4.8",
    "pyasn1-modules==0.2.8",
    "pyjwt==2.6.0",
    "pywin32==305",
    "pyyaml==6.0",
    "querystring-parser==1.2.4",
    "requests==2.28.1",
    "requests-oauthlib==1.3.1",
    "rsa==4.9",
    "shap==0.41.0",
    "sklearn==0.0.post1",
    "slicer==0.0.7",
    "smmap==5.0.0",
    "sqlalchemy==1.4.44",
    "sqlparse==0.4.3",
    "tabulate==0.9.0",
    "tensorboard==2.10.1",
    "tensorboard-data-server==0.6.1",
    "tensorboard-plugin-wit==1.8.1",
    "tensorflow==2.10.1",
    "tensorflow-estimator==2.10.0",
    "tensorflow-io-gcs-filesystem==0.27.0",
    "termcolor==2.1.0",
    "torch==1.13.0",
    "torchinfo==1.7.1",
    "torchvision==0.14.0",
    "tqdm==4.64.1",
    "typing-extensions==4.4.0",
    "tzdata==2022.6",
    "urllib3==1.26.12",
    "waitress==2.1.2",
    "websocket-client==1.4.2",
    "werkzeug==2.2.2",
    "wrapt==1.14.1",
    "zipp==3.10.0"
])"""

cd = CondaDependencies.create(pip_packages=[
    "azureml-defaults",
    "torch==1.12.0",
    "mlflow==2.0.1",
    "torchinfo==1.7.1",
    "torchvision==0.13.0",
    "cudatoolkit==11.3.1",
    "azureml-mlflow==1.48.0"
    ])

env.python.conda_dependencies = cd
# Specify a docker image to use.
env.docker.base_image = (
    "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04"
)

# Register environment to re-use later
env = env.register(workspace = ws)

datastore = ws.get_default_datastore()

# upload the data to the datastore
datastore.upload(src_dir=r'C:\Users\lbierling\OneDrive - KPMG\Projekte\Versicherung-Fehlererkennung\Project\image_recog_git\image_recog_git\insurance_image_recog\data3',
                 target_path='/data/',
                 overwrite=False,
                 show_progress=True)

from azureml.core import Dataset

# create the dataset object
dataset = Dataset.File.from_files(path=(datastore, '/data'))

# register the dataset for future use
dataset = dataset.register(workspace=ws,
                           name='crash_images',
                          description='Crash and no crash images for classification')

