from chalice import Chalice
from chalice import BadRequestError
import base64, os, boto3, ast
import numpy as np
import json
from urllib.request import urlopen 

app = Chalice(app_name='image-classifier')
app.debug=True

# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
sm_runtime = boto3.client('sagemaker-runtime')

@app.route('/', methods=['POST'])
def index():
    body = app.current_request.json_body
    if 'data' not in body:
        raise BadRequestError('Missing image data')
    if 'ENDPOINT_NAME' not in os.environ:
        raise BadRequestError('Missing endpoint')

    image = base64.b64decode(body['data']) # byte array
    endpoint = os.environ['ENDPOINT_NAME'] 

    # Invoking SageMaker Endpoint
    response = sm_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/x-image',
        Body=image)
                                          
    result = json.loads(response['Body'].read().decode())
    propa_result = "Result: " + str(np.argmax(result)) + "\n Probability: " + str(np.amax(result))

    
    # Load names for ImageNet classes
    text_url  = urlopen("https://raw.githubusercontent.com/a-shlash/chalice_app/master/imagenet1000_clsidx_to_labels.txt")

    object_categories = {}
    for line in text_url:
    	decoded_line = line.decode("utf-8")
    	key, val = decoded_line.strip().split(':')
    	object_categories[key] = val
    	
    propa_result = "Result: \n label: " + object_categories[str(np.argmax(result))]+ " \n probability: " + str(np.amax(result))
    return(propa_result)