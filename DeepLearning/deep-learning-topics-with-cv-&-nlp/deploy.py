import subprocess
import logging
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# Install smdebug if not installed
subprocess.call(['pip', 'install', 'smdebug'])

import smdebug

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    """
    Load the model from the model directory.

    Args:
        model_dir (str): Directory containing the model.

    Returns:
        model: The loaded model.
    """
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.required_grad = False
    
    num_features = model.fc.in_features
    num_classes = 133
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1)
    )
    
    model_path = os.path.join(model_dir, "model.pth")
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model

def input_fn(request_body, content_type):
    """
    Parse the request body and return the input object.

    Args:
        request_body (bytes): Request body containing image data.
        content_type (str): Content type of the request.

    Returns:
        input_object: Input object for prediction.
    """
    if content_type == 'image/jpeg':
        return Image.open(io.BytesIO(request_body))
    raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))

def predict_fn(input_object, model):
    """
    Perform prediction on the input object using the model.

    Args:
        input_object: Input object for prediction.
        model: Trained model for prediction.

    Returns:
        prediction: Prediction output.
    """
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    input_object = transform(input_object)
    
    with torch.no_grad():
        prediction = model(input_object.unsqueeze(0))
    return prediction
