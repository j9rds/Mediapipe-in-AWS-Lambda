# Lambda Image Processing
This code is designed to be deployed on AWS Lambda using lambda container images. It performs facial landmark detection and head orientation estimation on an input image. The output includes information about eye blink, head orientation, and mouth aspect ratio.

## Dependencies
The code requires the following dependencies:

 - mediapipe
 - numpy
 - cv2
 - boto3
 - uuid
 - json
## Usage
The entry point of the Lambda function is lambda_handler. The function takes an S3 image file key as input, performs image processing, and returns the output in JSON format.

## Input
The input event should be a dictionary with a single key-value pair. The key should be "queryStringParameters", and the value should be a dictionary with the following keys:

## Usage/Examples

"bucket" (optional): the name of the S3 bucket where the input image file is stored. If not provided, the default bucket name "action-validation" will be used.
"key": the key of the input image file in the S3 bucket.
Output
The output of the function is a dictionary with the following keys:


**orientation**: the estimated head orientation ("center", "left", or "right").

**orientation_confidence**: the confidence score of the estimated head orientation.

**re_closed**: a boolean value indicating whether the right eye is closed.

**le_closed**: a boolean value indicating whether the left eye is closed.

**re_confidence**: the confidence score of the right eye blink detection.

**le_confidence**: the confidence score of the left eye blink detection.

**re_ratio**: the right eye blink ratio.

**le_ratio**: the left eye blink ratio.

**eye_percent_difference**: the percent difference between the right and left eye blink ratios.

**mouth_closed**: a boolean value indicating whether the mouth is closed.

**mouth_confidence**: the confidence score of the mouth aspect ratio detection.

**hand_actions**: a list of actions detected in the image (if any). The list can be empty if no actions are detected.

## Example
Here is an example of the input event:

```json
{
    "queryStringParameters": {
        "bucket": "my-bucket",
        "key": "my-image.jpg"
    }
}
```
Here is an example of the output:


```json
{
    "orientation": "center",
    "orientation_confidence": 0.8,
    "re_closed": false,
    "le_closed": false,
    "re_confidence": 0.9,
    "le_confidence": 0.9,
    "re_ratio": 0.1,
    "le_ratio": 0.1,
    "eye_percent_difference": 0.0,
    "mouth_closed": false,
    "mouth_confidence": 0.9,
    "hand_actions": ["thumbs_up", "point_left"]
}
```