import yaml
import os
from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig, RunConfiguration
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azureml.core.runconfig import RunConfiguration
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
from azure.ai.ml.entities import ComputeInstance, AmlCompute, ComputeConfiguration


class AzureConnection:
    def __init__(self):
        with open("azure_config.yml") as y:
            self._config = yaml.safe_load(y)

        self._subscription_id = self._config["workspace"]["subscription_id"]
        self._resource_group = self._config["workspace"]["resource_group"]
        self._workspace_name = self._config["workspace"]["workspace_name"]
        self._create_workspace = self._config["workspace"]["create_workspace"]
        self._location = self._config["workspace"]["location"]
        self._compute_config_target = self._config["workspace"]["compute_config_target"]
        self._compute_name = self._config["workspace"]["compute_name"]
        self._vm_size = self._config["workspace"]["vm_size"]
        self._compute_target_name = self._config["workspace"]["compute_target"]
        self._experiment_name = self._config["workspace"]["experiment_name"]
        self._dataset_name = self._config["workspace"]["dataset_name"]
        self._dataset_id = self._config["workspace"]["dataset_id"]
        self._experiment_name = self._config["run_script"]["experiment_name"]
        self._run_source_directory = self._config["run_script"]["run_source_directory"]
        self._run_script_file_name = self._config["run_script"]["run_script_file_name"]
        self._create_environment = self._config["conda_environment"]["create_environment"]
        self._environment_name = self._config["conda_environment"]["environment_name"]
        self._pip_dependencies = self._config["conda_environment"]["pip_dependencies"]
        self._conda_dependencies = self._config["conda_environment"]["conda_dependencies"]
        self._upload_data = self._config["data_upload"]["upload"]
        self._path_to_data = self._config["data_upload"]["path_to_data"]
        self._target_path = self._config["data_upload"]["target_path"]
        self._upload_dataset_name = self._config["data_upload"]["dataset_name"]
        self._upload_dataset_description = self._config["data_upload"]["dataset_description"]

        self._compute_target = None
        self._workspace = None
        self._environment = None
        self._datastore = None
        self._dataset = None
        self._experiment = None

        if self._create_workspace:
            self.create_workspace()
        else:
            self.get_workspace()

        if self._create_environment:
            self.create_environment()
        else:
            self.get_environment()

        if self._upload_data:
            self.upload_data()
        else:
            self.get_dataset()

    def create_workspace(self):
        self._workspace = Workspace.create(name=self._workspace_name,
                                           subscription_id=self._subscription_id,
                                           resource_group=self._resource_group,
                                           location=self._location
                                           )

    def get_workspace(self):
        try:
            self._workspace = Workspace.from_config()
        except FileNotFoundError:
            self._workspace = Workspace(subscription_id=self._subscription_id,
                                        resource_group=self._resource_group,
                                        workspace_name=self._workspace_name,
                                        _location=self._location
                                        )

    def create_environment(self):
        self._environment = Environment(self._environment_name)
        cd = CondaDependencies.create(pip_packages=self._pip_dependencies)
        self._environment.python.conda_dependencies = cd
        self._environment.docker.base_image = ("mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.0.3-cudnn8-ubuntu18.04") # to be investigated !!
        self._environment = self._environment.register(workspace=self._workspace)
        self._environment.write_config(path=".azureml")

    def get_environment(self):
        if self._workspace is None:
            raise ValueError("No workspace configured!")

        self._environment = Environment.get(self._workspace, name=self._environment_name)

    def upload_data(self):
        datastore = self._workspace.get_default_datastore()

        # upload the data to the datastore
        datastore.upload(
            src_dir=self._path_to_data,
            target_path=self._target_path,
            overwrite=False,
            show_progress=True)
        dataset = Dataset.File.from_files(path=(datastore, '/data'))

        # register the dataset for future use
        self._dataset = dataset.register(workspace=self._workspace,
                                         name=self._upload_dataset_name,
                                         description=self._upload_dataset_description)

    def get_dataset(self):
        self._dataset = Dataset.get_by_name(self._workspace, self._dataset_name)

    def create_compute_instance(self):
        compute_config = ComputeConfiguration(vm_size=self._vm_size)
        compute_target = ComputeTarget.create(workspace=self._workspace, name=self._compute_name,
                                              provisioning_configuration=compute_config)
        compute_target.wait_for_completion(show_output=True)
        # Create a ComputeInstance object
        compute_instance = ComputeInstance(self._workspace, compute_target)

        # Wait for the instance to be created
        compute_instance.wait_for_completion(True)

    def run_script(self, use_data_uploaded_to_cloud=True):

        self._compute_target = self._workspace.compute_targets[self._compute_target_name]
        self._experiment = Experiment(workspace=self._workspace, name=self._experiment_name)

        if use_data_uploaded_to_cloud:
            # Be sure to add argparser to run script when using cloud data
            src = ScriptRunConfig(source_directory=self._run_source_directory,
                                  script=self._run_script_file_name,
                                  arguments=['--data-path', self._dataset.as_mount()],
                                  compute_target=self._compute_target,
                                  environment=self._environment)

        else:
            src = ScriptRunConfig(source_directory=self._run_source_directory,
                                  script=self._run_script_file_name,
                                  compute_target=self._compute_target,
                                  environment=self._environment)


        run = self._experiment.submit(config=src)








