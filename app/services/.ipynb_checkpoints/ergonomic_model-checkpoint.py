import os
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import json
from datetime import datetime
import warnings
from scipy.signal import savgol_filter, find_peaks
from scipy.stats import skew, kurtosis

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants
KEYPOINT_THRESHOLD = 0.3
SEQUENCE_LENGTH = 60
STRIDE = 30
MAX_GAP = 30
MODEL_PATH = "modelv4/reba_model.h5"
PREPROCESSING_PATH = "modelv4/preprocessing.joblib"

# Initialize model and resources
_model = None
_resources = None

def get_model_resources():
    """Load and return model resources, with lazy initialization"""
    global _model, _resources
    
    if _resources is not None:
        return _resources
    
    try:
        print("Loading ergonomic assessment model resources...")
        # Load model
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load preprocessing data
        preprocessing_data = joblib.load(PREPROCESSING_PATH)
        
        _resources = {
            'model': model,
            'scaler': preprocessing_data['scaler'],
            'model_features': preprocessing_data['model_features'],
            'core_angles': preprocessing_data.get('core_angles', [
                'Neck Angle', 'Left Upper Arm Angle', 'Right Upper Arm Angle',
                'Left Lower Arm Angle', 'Right Lower Arm Angle', 'Waist Angle',
                'Left Leg Angle', 'Right Leg Angle'
            ]),
            'sequence_length': preprocessing_data.get('sequence_length', SEQUENCE_LENGTH),
            'max_gap': preprocessing_data.get('max_gap', MAX_GAP),
            'static_window': preprocessing_data.get('static_window', 30),
            'imputation_columns': preprocessing_data.get('imputation_columns', 
                                                      ['Left Arm Imputed', 'Right Arm Imputed',
                                                       'Left Leg Imputed', 'Right Leg Imputed'])
        }
        
        print("Model resources loaded successfully")
        return _resources
        
    except Exception as e:
        print(f"Error loading model resources: {e}")
        return None

def engineer_features_for_single_image(row_dict, resources):
    """
    Engineer features for a single image using the same pipeline as training
    This matches your top 30 features exactly
    """
    if resources is None:
        from app.services.ergonomic_model import get_model_resources
        resources = get_model_resources()
        
    if resources is None:
        raise ValueError("Could not load model resources")
    
    # Create pseudo-sequence by replicating the single row (matches training approach)
    sequence_length = resources.get('sequence_length', 60)
    rows = [row_dict.copy() for _ in range(sequence_length)]
    
    # Add frame numbers to simulate temporal sequence
    for i, row in enumerate(rows):
        row['Frame'] = i
    
    df = pd.DataFrame(rows)
    
    # Core angles from your training
    core_angles = [
        'Neck Angle', 'Left Upper Arm Angle', 'Right Upper Arm Angle',
        'Left Lower Arm Angle', 'Right Lower Arm Angle', 'Waist Angle',
        'Left Leg Angle', 'Right Leg Angle'
    ]
    
    # 1. Basic transformations (matches your top 30 features)
    for angle in core_angles:
        if angle in df.columns:
            # Trigonometric features (sin, cos)
            df[f'{angle}_sin'] = np.sin(np.radians(df[angle]))
            df[f'{angle}_cos'] = np.cos(np.radians(df[angle]))
            
            # Squared features
            df[f'{angle}_squared'] = df[angle] ** 2
            
            # Log features (safe log with abs + 1)
            df[f'{angle}_log'] = np.log(np.abs(df[angle]) + 1)
    
    # 2. Range violation features (matches your training)
    normal_ranges = {
        'Neck Angle': (0, 45),
        'Waist Angle': (75, 105),
        'Left Upper Arm Angle': (-20, 120),
        'Right Upper Arm Angle': (-20, 120),
        'Left Lower Arm Angle': (60, 140),
        'Right Lower Arm Angle': (60, 140),
        'Left Leg Angle': (80, 120),
        'Right Leg Angle': (80, 120)
    }
    
    for angle, (min_val, max_val) in normal_ranges.items():
        if angle in df.columns:
            violations = (df[angle] < min_val) | (df[angle] > max_val)
            df[f'{angle}_range_violation'] = violations.astype(int)
    
    # 3. Slouch pattern (from your top 30)
    if 'Waist Angle' in df.columns and 'Neck Angle' in df.columns:
        slouch_pattern = (df['Waist Angle'] > 60) & (df['Neck Angle'] > 10)
        df['slouch_pattern'] = slouch_pattern.astype(float)
    
    # 4. Coordination dominance (from your top 30)
    df['coordination_dominance'] = 0.5  # Default value for single image
    
    # 5. Velocity and acceleration features (from your top 30)
    # For single image, these will be 0 but needed for model compatibility
    df['Waist Angle_velocity_mean'] = 0.0
    df['Waist Angle_acceleration_mean'] = 0.0
    
    # 6. Skewness feature (from your top 30)
    df['Left Lower Arm Angle_skewness'] = 0.0  # Default for single image
    
    # 7. Additional required features for compatibility
    additional_features = [
        'arm_coordination_symmetry', 'leg_coordination_symmetry', 
        'axial_coordination', 'cross_body_coordination', 'neck_waist_coupling',
        'com_stability', 'total_postural_sway', 'balance_challenge_index', 
        'stability_margin', 'movement_smoothness', 'movement_efficiency', 
        'coordination_consistency', 'joint_health_score', 'arm_ratio_consistency', 
        'natural_posture_score', 'postural_transition_density', 'movement_rhythmicity', 
        'postural_adaptation', 'postural_complexity', 'postural_entropy', 
        'forward_head_pattern', 'asymmetric_loading_pattern'
    ]
    
    # Set defaults for single image (no temporal data)
    for feature in additional_features:
        if feature not in df.columns:
            df[feature] = 0.5  # Neutral default
    
    # Add more skewness and kurtosis features that might be in model
    for angle in core_angles:
        if angle in df.columns:
            df[f'{angle}_skewness'] = 0.0  # Single image = no distribution
            df[f'{angle}_kurtosis'] = 0.0   # Single image = no distribution
    
    # Get model features (your top 30)
    model_features = resources['model_features']
    
    # Take the last row (most representative after temporal processing)
    final_row = df.iloc[-1]
    
    # Extract features that exist, set missing ones to 0
    features_dict = {}
    missing_features = []
    
    for feature in model_features:
        if feature in final_row:
            value = final_row[feature]
            # Ensure no NaN or inf values
            if np.isnan(value) or np.isinf(value):
                features_dict[feature] = 0.0
            else:
                features_dict[feature] = float(value)
        else:
            features_dict[feature] = 0.0
            missing_features.append(feature)
    
    if missing_features:
        print(f"Warning: Missing features set to 0: {missing_features[:5]}{'...' if len(missing_features) > 5 else ''}")
    
    # Calculate component scores using REBA scoring logic
    component_scores = calculate_component_scores_from_angles(row_dict)
    
    return features_dict, component_scores

def calculate_component_scores_from_angles(row_dict):
    """Calculate REBA component scores from joint angles (matches your REBA implementation)"""
    
    # Neck score calculation (matches your training)
    neck_angle = row_dict.get('Neck Angle', 0)
    if 0 <= neck_angle < 20:
        neck_score = 1
    elif 20 <= neck_angle < 45:
        neck_score = 2
    elif 45 <= neck_angle < 60:
        neck_score = 3
    elif neck_angle >= 60:
        neck_score = 4
    elif neck_angle < 0:
        neck_score = 2 if neck_angle >= -20 else 3
    else:
        neck_score = 1
    
    # Trunk score calculation (matches your training)
    waist_angle = row_dict.get('Waist Angle', 90)
    if waist_angle >= 0:  # Forward flexion
        if 0 <= waist_angle <= 5:
            trunk_score = 1
        elif 5 < waist_angle <= 20:
            trunk_score = 2
        elif 20 < waist_angle <= 60:
            trunk_score = 3
        elif 60 < waist_angle <= 90:
            trunk_score = 4
        elif 90 < waist_angle <= 120:
            trunk_score = 5
        else:
            trunk_score = 6
    else:  # Extension
        abs_angle = abs(waist_angle)
        if abs_angle <= 5:
            trunk_score = 1
        elif abs_angle <= 20:
            trunk_score = 2
        elif abs_angle < 45:
            trunk_score = 3
        else:
            trunk_score = 4
    
    # Upper arm score calculation (matches your training)
    left_upper = abs(row_dict.get('Left Upper Arm Angle', 0))
    right_upper = abs(row_dict.get('Right Upper Arm Angle', 0))
    max_upper_arm = max(left_upper, right_upper)
    
    if -20 <= max_upper_arm < 20:
        upper_arm_score = 1
    elif 20 <= max_upper_arm < 45:
        upper_arm_score = 2
    elif (max_upper_arm < -20) or (45 <= max_upper_arm < 90):
        upper_arm_score = 3
    elif max_upper_arm >= 90:
        upper_arm_score = 4
    else:
        upper_arm_score = 1
    
    # Lower arm score calculation (matches your training)
    def score_lower_arm(angle):
        return 1 if 60 <= angle < 100 else 2
    
    left_lower_score = score_lower_arm(row_dict.get('Left Lower Arm Angle', 90))
    right_lower_score = score_lower_arm(row_dict.get('Right Lower Arm Angle', 90))
    lower_arm_score = max(left_lower_score, right_lower_score)
    
    # Leg score calculation (kept for compatibility but expert said not used)
    def calc_leg_deviation(angle):
        return min(abs(angle - 90), abs(angle - 110))
    
    left_leg_dev = calc_leg_deviation(row_dict.get('Left Leg Angle', 100))
    right_leg_dev = calc_leg_deviation(row_dict.get('Right Leg Angle', 100))
    max_leg_dev = max(left_leg_dev, right_leg_dev)
    
    if max_leg_dev <= 20:
        leg_score = 2
    elif max_leg_dev <= 40:
        leg_score = 3
    elif max_leg_dev > 40:
        leg_score = 4
    else:
        leg_score = 1
    
    return {
        'trunk_score': int(trunk_score),
        'neck_score': int(neck_score),
        'upper_arm_score': int(upper_arm_score),
        'lower_arm_score': int(lower_arm_score),
        'leg_score': int(leg_score)  # Keep for compatibility
    }

def predict_single_image(features, resources=None):
    """Make prediction for a single image"""
    if resources is None:
        resources = get_model_resources()
        
    if resources is None:
        raise ValueError("Could not load model resources")
        
    model = resources['model']
    scaler = resources['scaler']
    
    # Convert features dict to array
    features_arr = np.array([list(features.values())])
    
    # Scale the features
    scaled_features = scaler.transform(features_arr)
    
    # For single image, repeat the features to create a sequence
    decay = np.linspace(1.0, 0.95, SEQUENCE_LENGTH)[:, np.newaxis]
    sequence = np.tile(scaled_features, (SEQUENCE_LENGTH, 1)) * decay
    
    # Predict
    prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)
    return float(prediction[0][0])

def predict_video(engineered_df, resources=None):
    """Make predictions for video sequences"""
    if resources is None:
        resources = get_model_resources()
        
    if resources is None:
        raise ValueError("Could not load model resources")
        
    model = resources['model']
    scaler = resources['scaler']
    model_features = resources['model_features']
    
    # Scale features
    scaled_features = scaler.transform(engineered_df[model_features])
    engineered_df[model_features] = scaled_features
    
    # Prepare sequences
    sequences = prepare_sequences(
        engineered_df, 
        model_features, 
        sequence_length=resources['sequence_length'],
        max_gap=resources['max_gap']
    )
    
    if sequences is None or len(sequences) == 0:
        print("⚠ No valid sequences found for prediction")
        return None
    
    # Predict
    predictions = np.array([
        model.predict(np.expand_dims(seq, axis=0), verbose=0).flatten()
        for seq in sequences
    ])
    
    # Return prediction statistics
    return {
        'predictions': predictions.flatten(),
        'average': float(np.mean(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'std': float(np.std(predictions))
    }

def prepare_sequences(df, model_features, sequence_length=SEQUENCE_LENGTH, stride=STRIDE, max_gap=MAX_GAP):
    """Prepare sequences for prediction"""
    sequences = []
    frames = df['Frame'].values
    data = df[model_features].values
    
    i = 0
    while i < len(df) - sequence_length + 1:
        seq_frames = frames[i:i+sequence_length]
        gaps = np.diff(seq_frames)
        
        if np.any(gaps > max_gap):
            bad_pos = np.where(gaps > max_gap)[0][0]
            i += bad_pos + 1
            continue
            
        sequences.append(data[i:i+sequence_length])
        i += stride
    
    return np.array(sequences) if sequences else None

def get_risk_level(reba_score):
    """Determine risk level from REBA score"""
    if reba_score <= 1:
        return "Negligible"
    elif reba_score <= 3:
        return "Low"
    elif reba_score <= 7:
        return "Medium"
    elif reba_score <= 10:
        return "High"
    else:
        return "Very High"

def get_action_level(reba_score):
    """Get action level based on REBA score"""
    if reba_score <= 1:
        return 0, "No action necessary"
    elif reba_score <= 3:
        return 1, "Action may be needed"
    elif reba_score <= 7:
        return 2, "Action necessary"
    elif reba_score <= 10:
        return 3, "Action necessary soon"
    else:
        return 4, "Action necessary NOW"

def generate_feedback(component_scores, reba_score):
    """Generate simple Indonesian feedback based on component scores and REBA score"""
    feedback = f"Skor REBA: {reba_score:.1f} - "
    
    # Add risk level in Indonesian
    risk_level = get_risk_level(reba_score)
    risk_level_id = {
        "Negligible": "Sangat Rendah",
        "Low": "Rendah", 
        "Medium": "Sedang",
        "High": "Tinggi",
        "Very High": "Sangat Tinggi"
    }.get(risk_level, risk_level)
    
    feedback += f"Risiko {risk_level_id}.\n\n"
    
    # Simple angle-based recommendations in Indonesian
    recommendations = []
    
    # Trunk recommendations based on component score
    if component_scores['trunk_score'] >= 3:
        recommendations.append("Luruskan punggung, jangan terlalu membungkuk.")
    
    # Neck recommendations  
    if component_scores['neck_score'] >= 2:
        recommendations.append("Angkat kepala, jangan terlalu menunduk.")
    
    # Upper arm recommendations
    if component_scores['upper_arm_score'] >= 3:
        recommendations.append("Turunkan posisi lengan atas, jangan terlalu terangkat.")
    
    # Lower arm recommendations
    if component_scores['lower_arm_score'] >= 2:
        recommendations.append("Atur sudut siku sekitar 90 derajat.")
    
    # General recommendations
    if reba_score <= 3:
        if not recommendations:
            recommendations.append("Postur sudah cukup baik, pertahankan posisi ini.")
    elif reba_score <= 7:
        if not recommendations:
            recommendations.append("Perbaiki postur duduk untuk mengurangi risiko.")
        recommendations.append("Sesekali ubah posisi untuk mengurangi kelelahan.")
    else:
        recommendations.append("Segera perbaiki postur duduk karena berisiko tinggi.")
        recommendations.append("Istirahat sejenak dan atur ulang posisi duduk.")
    
    # Add recommendations to feedback
    if recommendations:
        feedback += "Saran perbaikan:\n"
        for i, rec in enumerate(recommendations, 1):
            feedback += f"{i}. {rec}\n"
    
    return feedback

# Initialize model resources on import
get_model_resources()