import json

THRESHOLD = 0.85

def lambda_handler(event, context):
    """
    Lambda function to validate model inferences against a threshold.
    
    Args:
        event (dict): Lambda event containing the request body.
        context (object): Lambda context object.
        
    Returns:
        dict: Response containing image data, S3 bucket, S3 key, and model inferences.
        
    Raises:
        Exception: If threshold confidence is not met.
    """
    # Grab the inferences from the event
    inferences = event['body']['inferences']
    inferences = [float(value) for value in inferences[1:-1].split(',')]

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) >= THRESHOLD

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        "statusCode": 200,
        "body": {
            "image_data": event['body']['image_data'],
            "s3_bucket": event['body']['s3_bucket'],
            "s3_key": event['body']['s3_key'],
            "inferences": event['body']['inferences'],
        }
    }
