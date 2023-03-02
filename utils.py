import math
import cv2
from face_geometry import get_metric_landmarks, PCF, procrustes_landmark_basis
import numpy as np 
import mediapipe as mp

# Head orientation constants
frame_height, frame_width, channels = (720, 1280, 3)
# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)

camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

# indices of facial points
points_idx = [33, 263, 61, 291, 199]
points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
points_idx = list(set(points_idx))
points_idx.sort()

dist_coeff = np.zeros((4, 1))
horizontal_threshold = 0.5
vertical_threshold = 0.9
dict_index = {
    "forehead": 10,
    "chin": 152,
    "left cheek": 123,
    "right cheek": 352,
    "nose": 1,
    "bridge": 6,
}


mp_hands = mp.solutions.hands
def get_blink_Ratio(img, landmarks, right_indices, left_indices):
    """
    This function calculates the blink ratio of the eyes in a given image.

    Parameters:
    img (numpy.ndarray): An image represented as a numpy array.
    landmarks (list): A list of facial landmarks detected in the image.
    right_indices (list): A list of indices corresponding to the right eye landmarks.
    left_indices (list): A list of indices corresponding to the left eye landmarks.

    Returns:
    reRatio (float): The blink ratio for the right eye.
    leRatio: (float): The blink ratio for the left eye.
    """
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return reRatio, leRatio

def mouth_aspect_ratio(head_results, mouth_top, mouth_bottom, mouth_left, mouth_right):
    """
    This function calculates the mouth aspect ratio (MAR) using facial landmarks detected in an image.

    Parameters:
    head_results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList): The facial landmarks detected by the Mediapipe face detection model.
    mouth_top (int): The index of the landmark corresponding to the top point of the mouth.
    mouth_bottom (int): The index of the landmark corresponding to the bottom point of the mouth.
    mouth_left (int): The index of the landmark corresponding to the left point of the mouth.
    mouth_right (int): The index of the landmark corresponding to the right point of the mouth.

    Returns:
    MARValue (float): The mouth aspect ratio (MAR) value for the given facial landmarks.
    """
    landmark = head_results.multi_face_landmarks[0]

    topLip = landmark.landmark[mouth_top]
    bottomLip = landmark.landmark[mouth_bottom]
    left = landmark.landmark[mouth_left]
    right = landmark.landmark[mouth_right]

    vertical = euclaideanDistance((topLip.x, topLip.y), (bottomLip.x, bottomLip.y))
    horizontal = euclaideanDistance((left.x, left.y), (right.x, right.y))

    MARValue = horizontal / vertical

    return MARValue

def euclaideanDistance(point, point1):
    """
    This function calculates the Euclidean distance between two points in a 2D plane.

    Parameters:
    point (tuple): A tuple of (x, y) coordinates representing the first point.
    point1 (tuple): A tuple of (x, y) coordinates representing the second point.

    Returns:
    distance (float): The Euclidean distance between the two given points.
    """
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

def landmarksDetection(img, head_results, draw=False):
    """
    This function detects and returns the coordinates of facial landmarks in an image using the Mediapipe face detection model.

    Parameters:
    img (numpy.ndarray): An image represented as a numpy array.
    head_results (mediapipe.framework.formats.detection_pb2.Detection): The output of the Mediapipe face detection model.
    draw (bool): If True, the landmarks will be drawn on the image. Default is False.

    Returns:
    mesh_coord (list): A list of tuples, where each tuple contains the (x, y) coordinates of a facial landmark detected in the image.
    """
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [
        (int(point.x * img_width), int(point.y * img_height))
        for point in head_results.multi_face_landmarks[0].landmark
    ]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

def get_confidence(a, b, x):
    """
    This function calculates the confidence of a prediction based on a given input.

    Parameters:
    a (float): A value used in the confidence calculation formula.
    b (float): A value used in the confidence calculation formula.
    x (float): The input value used in the confidence calculation formula.

    Returns:
    confidence (float): The confidence score calculated based on the given input values.
    """
    confidence = ((1 / math.pi) * (abs(math.atan(a * (x - b))))) + 0.5
    return confidence

def get_head_orientation(landmarks):
    """
    Calculates the orientation of the head based on the 3D landmarks of the face.

    Parameters:
        landmarks (numpy.ndarray): array of shape (3, 468) containing the 3D landmarks of the face.

    Returns:
        orientation (list): a list containing the orientation of the head. Possible values are: ['up', 'down', 'left', 'right', 'center'].
        score (float): a score representing the confidence in the orientation estimation. The higher the score, the more confident the estimation.
    """
    # Define the parameters for the perspective camera transformation
    pcf = PCF(
        near=1,
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    # Get metric landmarks and transformation matrix
    metric_landmarks, pose_transform_mat = get_metric_landmarks(landmarks.copy(), pcf)

    # Extract the 3D model points and corresponding 2D image points for PnP algorithm
    model_points = metric_landmarks[0:3, points_idx].T
    image_points = (
        landmarks[0:2, points_idx].T * np.array([frame_width, frame_height])[None, :]
    )

    # Use solvePnP algorithm to estimate the rotation and translation vectors of the head
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    # Project the nose tip to get a point on the line of sight of the camera
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 25.0)]),
        rotation_vector,
        translation_vector,
        camera_matrix,
        dist_coeff,
    )

    
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    face_points_dict = {}
    for face_landmark in dict_index:
        face_index = dict_index[face_landmark]
        pos = [landmarks[0, face_index], landmarks[1, face_index]]
        face_points_dict[face_landmark] = pos
    # Calculate the horizontal and vertical distance between the nose and the line of sight
    rotation_vector_endpoint = (
        (nose_end_point2D[0][0][0]) / frame_width,
        (nose_end_point2D[0][0][1]) / frame_height,
    )
    
    # Initialize score and orientation list
    horizonal_distance = rotation_vector_endpoint[0] - face_points_dict["nose"][0]
    score = 1.0
    orientation = []

    # Check if the head is tilted to the left or right
    if horizonal_distance > 0:
        nose_cheek_x = abs(
            face_points_dict["right cheek"][0] - face_points_dict["nose"][0]
        )
        if abs(horizonal_distance * horizontal_threshold) > nose_cheek_x:
            score = face_points_dict["nose"][0] / face_points_dict["right cheek"][0]
            orientation.append("right")
    else:
        nose_cheek_x = abs(
            face_points_dict["left cheek"][0] - face_points_dict["nose"][0]
        )
        if abs(horizonal_distance * horizontal_threshold) > nose_cheek_x:
            score = face_points_dict["left cheek"][0] / face_points_dict["nose"][0]
            orientation.append("left")
    # Check if the head is tilted up or down
    vertical_distance = face_points_dict["nose"][1] - rotation_vector_endpoint[1]
    if vertical_distance > 0:
        dist = abs(face_points_dict["forehead"][1] - face_points_dict["nose"][1])
        if abs(vertical_distance * vertical_threshold) > dist:
            score = face_points_dict["bridge"][1] / face_points_dict["nose"][1]
            orientation.append("up")
    else:
        dist = abs(face_points_dict["chin"][1] - face_points_dict["nose"][1])
        if abs(vertical_distance * vertical_threshold) > dist:
            score = face_points_dict["nose"][1] / face_points_dict["chin"][1]
            orientation.append("down")
    # If the orientation list is empty append 'center'
    if len(orientation) == 0:
        orientation.append("center")
    return orientation, score

def countFingers(results):
    """
    This function will count the number of fingers up for each hand in the image.
    Args:
        results: The output of the hands landmarks detection performed on the image of the hands.
    Returns:
        fingers_statuses:   A dictionary containing the status (i.e., open or close) of each finger of both hands.
        count:              A dictionary containing the count of the fingers that are up, of both hands.
        hand_labels:        List containing labels of hands [left,right]
        hand_confidences:   Output score of mediapipe for hand presence
    """

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {"RIGHT": 0, "LEFT": 0}

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    # Initialize a dictionary to store the status (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {
        "RIGHT_THUMB": False,
        "RIGHT_INDEX": False,
        "RIGHT_MIDDLE": False,
        "RIGHT_RING": False,
        "RIGHT_PINKY": False,
        "LEFT_THUMB": False,
        "LEFT_INDEX": False,
        "LEFT_MIDDLE": False,
        "LEFT_RING": False,
        "LEFT_PINKY": False,
    }

    # Hand labels
    hand_labels = []
    hand_confidences = []
    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_label = hand_info.classification[0].label
        hand_labels.append(hand_label)
        # Retrieve hand confidence
        hand_confidence = hand_info.classification[0].score
        hand_confidences.append(hand_confidence)
        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Check if the finger is up by comparing the y-coordinates of the tip and pip landmarks.
            if (
                hand_landmarks.landmark[tip_index].y
                < hand_landmarks.landmark[tip_index - 2].y
            ):

                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper() + "_" + finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Retrieve the y-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and the x-coordinates of the retrieved landmarks.
        if (hand_label == "Right" and (thumb_tip_x < thumb_mcp_x)) or (
            hand_label == "Left" and (thumb_tip_x > thumb_mcp_x)
        ):

            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper() + "_THUMB"] = True

            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1

    # Check if the total count of the fingers of both hands are specified to be written on the output image.
    return fingers_statuses, count, hand_labels, hand_confidences

def get_hand_actions(fingers_statuses, hand_actions, hand_labels, hand_confidences):
    """
    Return left and right hand action from finger status
    fingers_statuses:   Dictionary contaning status of each finger
    hand_actions:       Dictionary from hand_actions.json
    hand_labels:        List containing the labels of hands
    """
    # Check if the hands are present
    hand_labels = [x.lower() for x in hand_labels]
    right_present = "right" in hand_labels
    left_present = "left" in hand_labels

    # Set the default actions to False
    right_pred_action = False
    left_pred_action = False
    right_confidence = False
    left_confidence = False

    # Get right hand actions
    if right_present:
        # Get confidence
        right_confidence = hand_confidences[hand_labels.index("right")]
        # Get the individual finger status
        right_fingers_status = [
            fingers_statuses[x] for x in fingers_statuses.keys() if "right" in x.lower()
        ]
        # Get the predicted action from hand_actions
        right_pred_actions = [
            k for k, v in hand_actions.items() if right_fingers_status == v
        ]
        if len(right_pred_actions) == 0:  # If not in the json
            right_pred_action = "No action set"
        else:
            right_pred_action = right_pred_actions[0]
    # Get left hand actions
    if left_present:
        # Get confidence
        left_confidence = hand_confidences[hand_labels.index("left")]
        # Get the individual finger status
        left_fingers_status = [
            fingers_statuses[x] for x in fingers_statuses.keys() if "left" in x.lower()
        ]
        # Get the predicted action from hand_actions
        left_pred_actions = [
            k for k, v in hand_actions.items() if left_fingers_status == v
        ]
        if len(left_pred_actions) == 0:  # If not in the json
            left_pred_action = "No action set"
        else:
            left_pred_action = left_pred_actions[0]

    return (
        right_present,
        left_present,
        right_pred_action,
        left_pred_action,
        right_confidence,
        left_confidence,
    )
