from transformers import AutoConfig, AutoModel

# Path to the directory containing the model files
model_directory = "mnist-digit-classification-2022-09-04"

# Load the model's configuration
config = AutoConfig.from_pretrained(model_directory)

# Load the PyTorch model
model = AutoModel.from_pretrained(model_directory, config=config)
