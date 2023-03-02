import copy
import itertools
import cv2
import mediapipe as mp
import math
import numpy as np
from face_geometry import procrustes_landmark_basis
import boto3
import uuid
import json
import os
from utils import landmarksDetection,get_blink_Ratio,get_head_orientation,get_confidence,mouth_aspect_ratio,countFingers,get_hand_actions

map_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
# For eyes constants
# Left eye indices
LEFT_EYE = [
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
]
# Right eye indices
RIGHT_EYE = [
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
]
# ------------------------------------------------------------
# Mouth Coordinates
MOUTH_COORDINATES = {
    "mouth_upper": 13,
    "mouth_lower": 14,
    "mouth_left": 78,
    "mouth_right": 308,
}
# Constants for confidence
eye_a = 8.2
mouth_a = 8.2

def lambda_handler(event, context):
    print(event)
    payload = event["queryStringParameters"]
    s3 = boto3.client("s3", region_name="ap-southeast-1")
    default_bucket = "action-validation"
    # Load Constants
    bucket = default_bucket if "bucket" not in payload else payload["bucket"]
    key = payload["key"]

    # Load hand_actions json
    s3_resource = boto3.resource("s3", region_name="ap-southeast-1")
    content_object = s3_resource.Object(default_bucket, "hand_actions.json")
    file_content = content_object.get()["Body"].read().decode("utf-8")
    hand_actions = json.loads(file_content)

    # Load eye constants
    eye_b = payload["eye_b"] if "eye_b" in payload else 4.5
    mouth_b = payload["mouth_b"] if "mouth_b" in payload else 3.0
    unique_id = str(uuid.uuid1())
    temp_img_path = f"/tmp/{unique_id}_{os.path.basename(key)}"
    s3.download_file(bucket, key, temp_img_path)
    output = {}

    with map_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True
    ) as face_mesh:
        # hand tracking
        hands = mp_hands.Hands(
            static_image_mode=True,
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )

        image = cv2.imread(temp_img_path)
        image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        # Convert color
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        head_results = face_mesh.process(rgb_image)
        # Process head results
        if head_results.multi_face_landmarks:
            mesh_coords = landmarksDetection(image, head_results, False)
            # Get blink ratio
            re_Ratio, le_Ratio = get_blink_Ratio(
                image, mesh_coords, LEFT_EYE, RIGHT_EYE
            )
            eye_percent_diff = abs(re_Ratio - le_Ratio) / (
                np.average([re_Ratio, le_Ratio])
            )
            # Head orientation
            head_landmarks = np.array(
                [
                    (lm.x, lm.y, lm.z)
                    for lm in head_results.multi_face_landmarks[0].landmark
                ]
            )[:468].T
            orientation, orientation_confidence = get_head_orientation(head_landmarks)
            output.update(
                {
                    "orientation": orientation,
                    "orientation_confidence": orientation_confidence,
                }
            )
            if orientation[0] == "center":
                # eyes closed or open
                re_closed = "True" if re_Ratio > eye_b else "False"
                le_closed = "True" if le_Ratio > eye_b else "False"

                # Calculate for confidence
                re_confidence = get_confidence(eye_a, eye_b, re_Ratio)
                le_confidence = get_confidence(eye_a, eye_b, le_Ratio)

                # Get mouth ratio
                MAR_ratio = mouth_aspect_ratio(
                    head_results,
                    MOUTH_COORDINATES["mouth_upper"],
                    MOUTH_COORDINATES["mouth_lower"],
                    MOUTH_COORDINATES["mouth_left"],
                    MOUTH_COORDINATES["mouth_right"],
                )
                mouth_closed = "True" if MAR_ratio > mouth_b else "False"
                mouth_confidence = get_confidence(mouth_a, mouth_b, MAR_ratio)

                # output
                output.update(
                    {
                        "re_closed": re_closed,
                        "le_closed": le_closed,
                        "re_confidence": re_confidence,
                        "le_confidence": le_confidence,
                        "re_ratio": re_Ratio,
                        "le_ratio": le_Ratio,
                        "eye_percent_difference": eye_percent_diff,
                        "mouth_closed": mouth_closed,
                        "mouth_confidence": mouth_confidence,
                    }
                )
        # Process hand results
        hand_results = hands.process(rgb_image)
        # dict_hands = {}
        rh_present, lh_present, rh_action, lh_action, rh_confidence, lh_confidence = [
            False
        ] * 6
        if hand_results.multi_hand_landmarks:
            fingers_statuses, count, hand_labels, hand_confidences = countFingers(
                hand_results
            )
            (
                rh_present,
                lh_present,
                rh_action,
                lh_action,
                rh_confidence,
                lh_confidence,
            ) = get_hand_actions(
                fingers_statuses, hand_actions, hand_labels, hand_confidences
            )

        output.update(
            {
                "rh_present": rh_present,
                "rh_confidence": round(rh_confidence, 2),
                "rh_action": rh_action,
                "lh_present": lh_present,
                "lh_confidence": round(lh_confidence, 2),
                "lh_action": lh_action,
            }
        )
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
            "Access-Control-Allow-Credentials": True,
            "Content-Type": "application/json",
        },
        "body": json.dumps(output),
    }


if __name__ == "__main__":
    event = {
        "queryStringParameters": {
            "key": "test/down.jpg",
            "bucket": "<insert bucket name>",
        }
    }
    result = lambda_handler(event, "test")
    print(result)
