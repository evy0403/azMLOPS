import argparse
from azureml.core import Workspace, Model, Environment, InferenceConfig
from azureml.core.webservice import AciWebservice

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
parser.add_argument("--endpoint_name", type=str, required=True, help="Name of the deployment endpoint")
args = parser.parse_args()

# Connect to Azure ML Workspace
ws = Workspace.from_config()

# Register the model
print("Registering the model...")
model = Model.register(workspace=ws, model_path=args.model_path, model_name="trained_model")
print(f"Model registered: {model.name}, Version: {model.version}")

# Create inference configuration
env = Environment.get(workspace=ws, name="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu")
inference_config = InferenceConfig(entry_script="scripts/score.py", environment=env)

# Deploy to ACI
print(f"Deploying model to endpoint: {args.endpoint_name}")
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
service = Model.deploy(ws, args.endpoint_name, [model], inference_config, aci_config)
service.wait_for_deployment(show_output=True)
print(f"Deployment complete. Scoring URI: {service.scoring_uri}")
