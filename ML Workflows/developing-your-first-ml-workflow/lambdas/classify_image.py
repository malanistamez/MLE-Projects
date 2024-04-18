import json
import base64

from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-04-15-05-38-02-192"

def lambda_handler(event, context):
    """
    Lambda function to handle image classification requests.
    
    Args:
        event (dict): Lambda event containing the request body.
        context (object): Lambda context object.
        
    Returns:
        dict: Response containing image data, S3 bucket, S3 key, and model inferences.
    """
    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])
    
    # Instantiate a Predictor
    predictor = Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer('image/png')
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    # We return the data back to the Step Function    
    event['inferences'] = inferences.decode('utf-8')
    
    return {
        "statusCode": 200,
        "body": {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['inferences'],
        }
    }
