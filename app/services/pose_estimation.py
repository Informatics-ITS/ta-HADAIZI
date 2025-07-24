import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path
from app.services.ergonomic_model import get_model_resources, predict_single_image, get_risk_level, generate_feedback
from app.services.image_visualizer import generate_pose_visualization

# MoveNet model initialization
_movenet = None
_input_size = 256

# Constants
KEYPOINT_THRESHOLD = 0.3
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3,
    'right_ear': 4, 'left_shoulder': 5, 'right_shoulder': 6, 
    'left_elbow': 7, 'right_elbow': 8, 'left_wrist': 9,
    'right_wrist': 10, 'left_hip': 11, 'right_hip': 12, 
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

def load_movenet_from_cache(model_name="movenet_lightning", cache_dir="movenet_models"):
    """Load MoveNet from local cache directory"""
    cache_path = Path(cache_dir)
    model_path = cache_path / model_name
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"Please download the model first using the MoveNet downloader")
        return None, None
    
    try:
        print(f"📁 Loading {model_name} from cache: {model_path}")
        
        # Load the model from the cached directory
        module = tf.saved_model.load(str(model_path))
        
        def movenet_wrapper(input_image):
            """Runs MoveNet on an input image."""
            model = module.signatures['serving_default']
            input_image = tf.cast(input_image, dtype=tf.int32)
            outputs = model(input_image)
            keypoints_with_scores = outputs['output_0'].numpy()
            return keypoints_with_scores
        
        # Set input size based on model type
        input_size = 192 if "lightning" in model_name else 256
        
        print(f"✅ {model_name} loaded successfully! Input size: {input_size}")
        return movenet_wrapper, input_size
        
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        print("💡 Try downloading the model again")
        return None, None

def init_movenet():
    """Initialize the MoveNet model from local cache"""
    global _movenet, _input_size
    
    if _movenet is not None:
        return _movenet, _input_size
    
    try:
        # Try to load from cache first
        _movenet, _input_size = load_movenet_from_cache('movenet_lightning', 'movenet_models')
        
        if _movenet is None:
            print("❌ Failed to load from cache, falling back to TensorFlow Hub...")
            # Fallback to TensorFlow Hub if cache loading fails
            import tensorflow_hub as hub
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"
            module = hub.load(model_url)
            
            def movenet_wrapper(input_image):
                """Runs MoveNet on an input image."""
                model = module.signatures['serving_default']
                input_image = tf.cast(input_image, dtype=tf.int32)
                outputs = model(input_image)
                keypoints_with_scores = outputs['output_0'].numpy()
                return keypoints_with_scores
            
            _movenet = movenet_wrapper
            _input_size = 256
            print("✅ MoveNet loaded from TensorFlow Hub")
        
        return _movenet, _input_size
        
    except Exception as e:
        print(f"❌ Error initializing MoveNet: {e}")
        print("💡 Make sure the movenet_models directory contains the downloaded model")
        return None, None

class AngleSmoother:
    """Helper class to smooth angle measurements"""
    def __init__(self, window_size=3):
        self.history = deque(maxlen=window_size)
        
    def smooth(self, angle):
        if angle is not None:
            self.history.append(angle)
            if len(self.history) > 0:
                return np.mean(self.history)
        return None

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    
    return angle

def init_crop_region(image_height, image_width):
    """Defines the default crop region."""
    if image_width > image_height:
        box_height = image_width / image_height
        box_width = 1.0
        y_min = (image_height / 2 - image_width / 2) / image_height
        x_min = 0.0
    else:
        box_height = 1.0
        box_width = image_height / image_width
        y_min = 0.0
        x_min = (image_width / 2 - image_height / 2) / image_width

    return {
        'y_min': y_min,
        'x_min': x_min,
        'y_max': y_min + box_height,
        'x_max': x_min + box_width,
        'height': box_height,
        'width': box_width
    }

def crop_and_resize(image, crop_region, crop_size):
    """Crops and resizes the image to prepare for the model input."""
    boxes = [[crop_region['y_min'], crop_region['x_min'],
             crop_region['y_max'], crop_region['x_max']]]
    output_image = tf.image.crop_and_resize(
        image, box_indices=[0], boxes=boxes, crop_size=crop_size)
    return output_image

def should_flip_image(keypoints_with_scores):
    """Determines if the image should be flipped based on keypoint positions."""
    # Get relevant keypoints with confidence checks
    left_shoulder = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_shoulder']]
    right_shoulder = keypoints_with_scores[0, 0, KEYPOINT_DICT['right_shoulder']]
    left_wrist = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_wrist']]
    left_knee = keypoints_with_scores[0, 0, KEYPOINT_DICT['left_knee']]

    score = 0
    valid_keypoints = 0
    
    # Shoulder comparison
    if left_shoulder[2] > KEYPOINT_THRESHOLD and right_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_shoulder[1] > right_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    # Wrist position
    if left_wrist[2] > KEYPOINT_THRESHOLD and left_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_wrist[1] < left_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    # Knee position
    if left_knee[2] > KEYPOINT_THRESHOLD and left_shoulder[2] > KEYPOINT_THRESHOLD:
        if left_knee[1] < left_shoulder[1]:
            score += 1  # Facing left
        else:
            score -= 1  # Facing right
        valid_keypoints += 1
    
    return score > 0 if valid_keypoints >= 2 else False

def run_inference(movenet, image, crop_region, crop_size):
    """Runs model inference on the cropped region with proper flip handling."""
    image_height, image_width, _ = image.shape
    
    # First pass to determine orientation
    input_image = crop_and_resize(tf.expand_dims(image, axis=0), crop_region, crop_size=crop_size)
    keypoints_with_scores = movenet(input_image)
    flip_required = should_flip_image(keypoints_with_scores)
    
    # Second pass if flipping is needed
    if flip_required:
        flipped_image = cv2.flip(image, 1)
        input_image = crop_and_resize(tf.expand_dims(flipped_image, axis=0), crop_region, crop_size=crop_size)
        keypoints_with_scores = movenet(input_image)
        original_image = image.copy()
        image = flipped_image
    else:
        original_image = image.copy()
    
    # Adjust keypoints for crop region
    for idx in range(17):
        keypoints_with_scores[0, 0, idx, 0] = (
            crop_region['y_min'] * image_height +
            crop_region['height'] * image_height *
            keypoints_with_scores[0, 0, idx, 0]) / image_height
        keypoints_with_scores[0, 0, idx, 1] = (
            crop_region['x_min'] * image_width +
            crop_region['width'] * image_width *
            keypoints_with_scores[0, 0, idx, 1]) / image_width

    return keypoints_with_scores, image, original_image, flip_required

def get_keypoint_if_valid(validated_keypoints, keypoint_name):
    """Get a valid keypoint if it exists"""
    kp = validated_keypoints[keypoint_name]
    return (kp['y'], kp['x']) if kp['valid'] else None

def calculate_angle_with_fallback(a_name, b_name, c_name, angle_name, validated_keypoints, imputed_angles, neutral_angles):
    """Calculate angle with fallback to neutral angles"""
    a = get_keypoint_if_valid(validated_keypoints, a_name)
    b = get_keypoint_if_valid(validated_keypoints, b_name)
    c = get_keypoint_if_valid(validated_keypoints, c_name)
    
    if a is not None and b is not None and c is not None:
        try:
            angle = calculate_angle(a, b, c)
            return angle
        except:
            pass
    
    # If we get here, use neutral angle and flag as imputed
    imputed_angles[angle_name] = True
    return neutral_angles[angle_name]

def get_joint_angles(keypoints_with_scores, keypoint_threshold=KEYPOINT_THRESHOLD):
    """
    Calculate joint angles from pose keypoints with enhanced debugging
    
    Args:
        keypoints_with_scores: Output from MoveNet model
        keypoint_threshold: Confidence threshold for valid keypoints
    
    Returns:
        dict: Contains angles and imputation flags
    """
    keypoints = keypoints_with_scores[0, 0, :, :2]
    scores = keypoints_with_scores[0, 0, :, 2]

    # Initialize smoothers if they don't exist
    if not hasattr(get_joint_angles, 'smoothers'):
        get_joint_angles.smoothers = {
            'left_leg': AngleSmoother(),
            'right_leg': AngleSmoother(),
            'neck': AngleSmoother(),
            'trunk': AngleSmoother(),
            'upper_arm': AngleSmoother(),
            'lower_arm': AngleSmoother(),
        }

    # Initialize tracking dictionaries
    imputed_angles = {
        'left_leg': False,
        'right_leg': False,
        'neck': False,
        'waist': False,
        'left_upper_arm': False,
        'right_upper_arm': False,
        'left_lower_arm': False,
        'right_lower_arm': False
    }

    neutral_angles = {
        'left_leg': 100,
        'right_leg': 100,
        'left_upper_arm': 0,
        'right_upper_arm': 0,
        'left_lower_arm': 90,
        'right_lower_arm': 90,
        'waist': 110,
        'neck': 5
    }

    # Create validated keypoints dictionary
    validated_keypoints = {}
    for name, idx in KEYPOINT_DICT.items():
        validated_keypoints[name] = {
            'x': keypoints[idx][1] if scores[idx] > keypoint_threshold else None,
            'y': keypoints[idx][0] if scores[idx] > keypoint_threshold else None,
            'valid': scores[idx] > keypoint_threshold,
            'confidence': float(scores[idx])  # Add confidence for debugging
        }

    # Calculate all angles with fallback
    angles = {}
    
    # === WAIST ANGLE CALCULATION WITH DEBUGGING ===
    shoulder_left = get_keypoint_if_valid(validated_keypoints, 'left_shoulder')
    shoulder_right = get_keypoint_if_valid(validated_keypoints, 'right_shoulder')
    hip_left = get_keypoint_if_valid(validated_keypoints, 'left_hip')
    hip_right = get_keypoint_if_valid(validated_keypoints, 'right_hip')
    
    # Track which keypoints are missing for waist calculation
    waist_missing_keypoints = []
    waist_keypoint_confidences = {}
    
    if shoulder_left is None:
        waist_missing_keypoints.append('left_shoulder')
    else:
        waist_keypoint_confidences['left_shoulder'] = validated_keypoints['left_shoulder']['confidence']
        
    if shoulder_right is None:
        waist_missing_keypoints.append('right_shoulder')
    else:
        waist_keypoint_confidences['right_shoulder'] = validated_keypoints['right_shoulder']['confidence']
        
    if hip_left is None:
        waist_missing_keypoints.append('left_hip')
    else:
        waist_keypoint_confidences['left_hip'] = validated_keypoints['left_hip']['confidence']
        
    if hip_right is None:
        waist_missing_keypoints.append('right_hip')
    else:
        waist_keypoint_confidences['right_hip'] = validated_keypoints['right_hip']['confidence']
    
    if all([shoulder_left, shoulder_right, hip_left, hip_right]):
        # Original angle calculation
        shoulder_vec = np.array([shoulder_left[0] - shoulder_right[0],
                                 shoulder_left[1] - shoulder_right[1]])
        hip_vec = np.array([hip_left[0] - hip_right[0],
                            hip_left[1] - hip_right[1]])
        
        dot_product = np.dot(shoulder_vec, hip_vec)
        shoulder_mag = np.linalg.norm(shoulder_vec)
        hip_mag = np.linalg.norm(hip_vec)
        
        if shoulder_mag > 0 and hip_mag > 0:
            cos_angle = dot_product / (shoulder_mag * hip_mag)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            unsigned_angle = np.degrees(np.arccos(cos_angle))
            
            # Trunk flexion (forward/backward lean)
            shoulder_center_y = (shoulder_left[1] + shoulder_right[1]) / 2
            hip_center_y = (hip_left[1] + hip_right[1]) / 2
            
            # Forward lean (shoulders lower than hips in image coordinates)
            if shoulder_center_y > hip_center_y:
                angles['waist'] = unsigned_angle  # Positive for forward
                angles['waist_direction'] = "forward"
            else:
                angles['waist'] = -unsigned_angle  # Negative for backward
                angles['waist_direction'] = "backward"
            
            imputed_angles['waist'] = False
        else:
            angles['waist'] = neutral_angles['waist']
            imputed_angles['waist'] = True
    else:
        angles['waist'] = neutral_angles['waist']
        imputed_angles['waist'] = True

    # === NECK ANGLE CALCULATION WITH DEBUGGING ===
    ear_point = get_keypoint_if_valid(validated_keypoints, 'left_ear')
    neck_missing_keypoints = []
    neck_keypoint_confidences = {}
    
    # Check left ear
    if ear_point is None:
        neck_missing_keypoints.append('left_ear')
        # Try right ear
        ear_point = get_keypoint_if_valid(validated_keypoints, 'right_ear')
        if ear_point is None:
            neck_missing_keypoints.append('right_ear')
        else:
            neck_keypoint_confidences['right_ear'] = validated_keypoints['right_ear']['confidence']
    else:
        neck_keypoint_confidences['left_ear'] = validated_keypoints['left_ear']['confidence']
    
    # Check shoulders for neck calculation
    if shoulder_left is None:
        if 'left_shoulder' not in neck_missing_keypoints:
            neck_missing_keypoints.append('left_shoulder')
    else:
        neck_keypoint_confidences['left_shoulder'] = validated_keypoints['left_shoulder']['confidence']
        
    if shoulder_right is None:
        if 'right_shoulder' not in neck_missing_keypoints:
            neck_missing_keypoints.append('right_shoulder')
    else:
        neck_keypoint_confidences['right_shoulder'] = validated_keypoints['right_shoulder']['confidence']
    
    if ear_point is not None and shoulder_left is not None and shoulder_right is not None:
        # Calculate midpoint between shoulders
        mid_shoulder = ((shoulder_left[0] + shoulder_right[0])/2, 
                       (shoulder_left[1] + shoulder_right[1])/2)
        
        # Calculate angle between ear and mid-shoulder point (vertical line)
        # Create a point directly above mid_shoulder (same x, lower y in image coordinates)
        vertical_point = (mid_shoulder[0] - 1, mid_shoulder[1])
        
        try:
            angle = calculate_angle(ear_point, mid_shoulder, vertical_point)
            angles['neck'] = angle
            imputed_angles['neck'] = False
        except:
            angles['neck'] = neutral_angles['neck']
            imputed_angles['neck'] = True
    else:
        angles['neck'] = neutral_angles['neck']
        imputed_angles['neck'] = True

    # Calculate other angles (unchanged)
    angle_mapping = {
        'left_upper_arm': ('left_hip', 'left_shoulder', 'left_elbow'),
        'right_upper_arm': ('right_hip', 'right_shoulder', 'right_elbow'),
        'left_lower_arm': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_lower_arm': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_leg': ('left_hip', 'left_knee', 'left_ankle'),
        'right_leg': ('right_hip', 'right_knee', 'right_ankle')
    }

    for angle_name, points in angle_mapping.items():
        angles[angle_name] = calculate_angle_with_fallback(
            points[0], points[1], points[2], angle_name,
            validated_keypoints, imputed_angles, neutral_angles)

    # Apply smoothing
    for angle_name in angles:
        if angle_name in get_joint_angles.smoothers:
            angles[angle_name] = get_joint_angles.smoothers[angle_name].smooth(angles[angle_name])

    # === ENHANCED DEBUGGING FOR MISSING ANGLES ===
    has_minimum_angles = not imputed_angles['neck'] and not imputed_angles['waist']
    
    if not has_minimum_angles:
        missing_angles = []
        debug_info = []
        
        if imputed_angles['neck']:
            missing_angles.append("neck")
            if neck_missing_keypoints:
                # Get confidence values for missing neck keypoints
                neck_low_conf = []
                for kp in neck_missing_keypoints:
                    conf = validated_keypoints[kp]['confidence']
                    neck_low_conf.append(f"{kp}:{conf:.3f}")
                debug_info.append(f"NECK missing keypoints: {', '.join(neck_low_conf)}")
        
        if imputed_angles['waist']:
            missing_angles.append("waist")
            if waist_missing_keypoints:
                # Get confidence values for missing waist keypoints
                waist_low_conf = []
                for kp in waist_missing_keypoints:
                    conf = validated_keypoints[kp]['confidence']
                    waist_low_conf.append(f"{kp}:{conf:.3f}")
                debug_info.append(f"WAIST missing keypoints: {', '.join(waist_low_conf)}")
        
        print(f"Skipping frame - Missing angles: {', '.join(missing_angles)}")
        for info in debug_info:
            print(f"  {info}")
        
        return None

    # Also store imputation information
    angles['imputed_angles'] = imputed_angles
    
    return angles

def create_row_dict(angles, filename, frame_num):
    """Create a dictionary representing one row of data"""
    if angles is None:
        return None
        
    imputed_angles = angles.get('imputed_angles', {})
    
    row = {
        'File Name': filename,
        'Frame': frame_num,
        
        # Core Angles
        'Neck Angle': angles.get('neck', -1),
        'Left Upper Arm Angle': angles.get('left_upper_arm', -1),
        'Right Upper Arm Angle': angles.get('right_upper_arm', -1),
        'Left Lower Arm Angle': angles.get('left_lower_arm', -1),
        'Right Lower Arm Angle': angles.get('right_lower_arm', -1),
        'Waist Angle': angles.get('waist', -1),
        'Left Leg Angle': angles.get('left_leg', -1),
        'Right Leg Angle': angles.get('right_leg', -1),
        
        # Imputation Flags
        'Left Arm Imputed': int(imputed_angles.get('left_upper_arm', False)),
        'Right Arm Imputed': int(imputed_angles.get('right_upper_arm', False)),
        'Left Leg Imputed': int(imputed_angles.get('left_leg', False)),
        'Right Leg Imputed': int(imputed_angles.get('right_leg', False)),
    }
    
    return row

def process_pose_from_bytes(image_bytes, output_visualization=True):
    """
    Process an image from bytes, detect pose, and generate predictions
    """
    try:
        # Initialize MoveNet
        movenet, input_size = init_movenet()
        if movenet is None:
            raise ValueError("Could not initialize MoveNet model")
            
        # Load model resources
        resources = get_model_resources()
        if resources is None:
            raise ValueError("Could not load model resources")
            
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image data")
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose detection
        keypoints, processed_img, original_img, flip_required = run_inference(
            movenet, frame, init_crop_region(frame.shape[0], frame.shape[1]), 
            crop_size=[input_size, input_size])
        
        # Calculate joint angles
        angles = get_joint_angles(keypoints)
        if angles is None:
            raise ValueError("Insufficient keypoints detected in image")
        
        # Create row dictionary
        filename = f"image_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        row_dict = create_row_dict(angles, filename, 0)
        
        # Engineer features and predict using the updated pipeline
        from app.services.ergonomic_model import engineer_features_for_single_image
        features, component_scores = engineer_features_for_single_image(row_dict, resources)
        
        # Make prediction
        reba_score = predict_single_image(features, resources)
        
        # Generate feedback
        feedback = generate_feedback(component_scores, reba_score)
        
        # Add REBA score to component scores for visualization
        component_scores['reba_score'] = reba_score
        
        # Initialize link variable
        link_image = None
        
        if output_visualization:
            # Generate visualization
            visualization = generate_pose_visualization(
                processed_img, keypoints, component_scores, original_img, flip_required,
                angle_values=angles
            )
            
            # Save visualization - EXACTLY like your friend's approach
            link_image = save_visualization_like_friend(visualization, component_scores)
        
        # Store actual angle values
        angle_values = {
            'neck': float(angles['neck']),
            'waist': float(angles['waist']),
            'left_upper_arm': float(angles['left_upper_arm']),
            'right_upper_arm': float(angles['right_upper_arm']),
            'left_lower_arm': float(angles['left_lower_arm']),
            'right_lower_arm': float(angles['right_lower_arm']),
            'left_leg': float(angles['left_leg']),
            'right_leg': float(angles['right_leg'])
        }
        
        # Create result dictionary
        result = {
            'reba_score': float(reba_score),
            'risk_level': get_risk_level(reba_score),
            'component_scores': component_scores,
            'angle_values': angle_values,
            'feedback': feedback
        }
        
        # Add link if visualization was created
        if link_image:
            result['link_image'] = link_image
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def save_visualization_like_friend(visualization, hasil_prediksi):
    """
    Save visualization exactly like your friend's approach
    """
    # Convert RGB to BGR for OpenCV
    img = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    
    # Create folder path exactly like your friend
    folder_path = os.path.join("output_images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(folder_path, exist_ok=True)
    
    # Create filename exactly like your friend
    filename = datetime.now().strftime("%H%M%S_%f") + "_hasil.png"
    filepath = os.path.join(folder_path, filename)
    
    # Create link exactly like your friend (only change: model1 -> model2)
    link_image = "https://vps.danar.site/model2/" + filepath
    
    # Save the image
    cv2.imwrite(filepath, img)
    
    return link_image