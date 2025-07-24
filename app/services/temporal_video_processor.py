import numpy as np
import os
from collections import deque
from tensorflow.keras.models import load_model

class TemporalProcessor:
    """
    Temporal LSTM processor for video analysis
    Handles 60-frame sequence prediction matching your training methodology
    """
    
    def __init__(self, sequence_length=60, stride=30, max_gap=90):
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_gap = max_gap
        self.temporal_model = None
        
    def load_temporal_model(self, model_path="modelv4/reba_model.h5"):
        """Load the trained temporal BiLSTM model"""
        try:
            if os.path.exists(model_path):
                self.temporal_model = load_model(model_path)
                print(f"Temporal BiLSTM model loaded from {model_path}")
                return True
            else:
                print(f"Temporal model not found at {model_path}")
                return False
        except Exception as e:
            print(f"Error loading temporal model: {e}")
            return False
    
    def create_temporal_sequences(self, rows):
        """
        Create temporal sequences from video frame data
        Matches your training gap-aware methodology
        """
        if len(rows) < self.sequence_length:
            print(f"Warning: Video too short ({len(rows)} frames), need {self.sequence_length} minimum")
            return [], []
        
        sequences = []
        sequence_info = []
        
        # Sort rows by frame number
        sorted_rows = sorted(rows, key=lambda x: x['Frame'])
        
        # Create sliding window sequences
        for i in range(0, len(sorted_rows) - self.sequence_length + 1, self.stride):
            sequence_rows = sorted_rows[i:i+self.sequence_length]
            
            # Check for gaps (like your training gap-aware processing)
            frame_numbers = [row['Frame'] for row in sequence_rows]
            gaps = [frame_numbers[j+1] - frame_numbers[j] for j in range(len(frame_numbers)-1)]
            max_gap_found = max(gaps) if gaps else 0
            
            if max_gap_found > self.max_gap:
                print(f"   Skipping sequence starting at frame {frame_numbers[0]}: large gap detected ({max_gap_found})")
                continue
            
            sequences.append(sequence_rows)
            sequence_info.append({
                'start_frame': frame_numbers[0],
                'end_frame': frame_numbers[-1],
                'sequence_index': len(sequences) - 1,
                'max_gap': max_gap_found
            })
        
        return sequences, sequence_info
    
    def predict_temporal_sequence(self, sequence_rows, resources):
        """
        Predict REBA score using temporal LSTM for a 60-frame sequence
        """
        try:
            # Import here to avoid circular imports
            from app.services.ergonomic_model import engineer_features_for_single_image, predict_single_image
            
            # Extract features for each frame in sequence
            sequence_features = []
            for row in sequence_rows:
                features, _ = engineer_features_for_single_image(row, resources)
                sequence_features.append(features)
            
            # Convert to numpy array and reshape for LSTM input
            sequence_array = np.array(sequence_features)  # Shape: (60, 30)
            sequence_input = sequence_array.reshape(1, self.sequence_length, -1)  # Shape: (1, 60, 30)
            
            # Use temporal model if available
            if self.temporal_model is not None:
                # Predict using temporal context
                prediction = self.temporal_model.predict(sequence_input, verbose=0)
                return float(prediction[0][0])
            else:
                # Fallback to single frame prediction of last frame
                print("Warning: Temporal model not available, using single frame prediction")
                return predict_single_image(sequence_features[-1], resources)
                
        except Exception as e:
            print(f"Error in temporal prediction: {e}")
            # Fallback to single frame prediction of last frame
            from app.services.ergonomic_model import predict_single_image
            return predict_single_image(sequence_features[-1], resources)
    
    def analyze_temporal_predictions(self, rows, resources):
        """
        Complete temporal analysis of video frames
        Returns both single-frame and temporal predictions with comparison
        """
        if not rows:
            return None
            
        print(f"Starting temporal analysis of {len(rows)} frames...")
        
        # Create temporal sequences
        sequences, sequence_info = self.create_temporal_sequences(rows)
        
        if not sequences:
            print("No valid temporal sequences created")
            return None
        
        print(f"Created {len(sequences)} temporal sequences")
        
        # Predict for each temporal sequence
        temporal_predictions = []
        for i, (sequence, info) in enumerate(zip(sequences, sequence_info)):
            temporal_pred = self.predict_temporal_sequence(sequence, resources)
            temporal_predictions.append({
                'sequence_index': i,
                'start_frame': info['start_frame'],
                'end_frame': info['end_frame'],
                'temporal_reba_score': temporal_pred,
                'max_gap': info['max_gap']
            })
            print(f"   Sequence {i+1}: frames {info['start_frame']}-{info['end_frame']} -> REBA {temporal_pred:.3f}")
        
        # Calculate temporal statistics
        temporal_scores = [p['temporal_reba_score'] for p in temporal_predictions]
        temporal_analysis = {
            'temporal_predictions': temporal_predictions,
            'avg_temporal_reba': np.mean(temporal_scores),
            'min_temporal_reba': np.min(temporal_scores),
            'max_temporal_reba': np.max(temporal_scores),
            'temporal_sequences_count': len(temporal_predictions),
            'sequence_length_used': self.sequence_length,
            'stride_used': self.stride,
            'max_gap_used': self.max_gap
        }
        
        print(f"Temporal analysis complete:")
        print(f"   Average temporal REBA: {temporal_analysis['avg_temporal_reba']:.3f}")
        print(f"   Range: {temporal_analysis['min_temporal_reba']:.3f} - {temporal_analysis['max_temporal_reba']:.3f}")
        
        return temporal_analysis


def enhance_resources_with_temporal_model(resources, model_path="modelv4/reba_model.h5"):
    """
    Add temporal model to existing resources
    Call this in your get_model_resources function
    """
    temporal_processor = TemporalProcessor()
    model_loaded = temporal_processor.load_temporal_model(model_path)
    
    resources['temporal_processor'] = temporal_processor
    resources['has_temporal_model'] = model_loaded
    
    return resources


def create_enhanced_frame_results(rows, resources):
    """
    Enhanced frame results creation with temporal LSTM prediction
    This replaces your create_frame_results function
    """
    from app.services.ergonomic_model import engineer_features_for_single_image, predict_single_image
    from app.utils.summarize_results import summarize_results
    from app.services.ergonomic_model import get_risk_level
    import pandas as pd
    
    print(f"Creating enhanced results with temporal analysis...")
    
    # First, create single-frame results (your existing approach)
    single_frame_results = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i+batch_size]
        df_batch = pd.DataFrame(batch_rows)
        
        for _, row in df_batch.iterrows():
            row_dict = row.to_dict()
            
            try:
                # Use the existing feature engineering pipeline
                features, component_scores = engineer_features_for_single_image(row_dict, resources)
                # Make prediction using the existing model
                reba_score = predict_single_image(features, resources)
                
                # Create result with existing structure
                result = {
                    'frame': int(row['Frame']),
                    'reba_score': float(reba_score),
                    'component_scores': {
                        'trunk': int(component_scores['trunk_score']),
                        'neck': int(component_scores['neck_score']),
                        'upper_arm': int(component_scores['upper_arm_score']),
                        'lower_arm': int(component_scores['lower_arm_score']),
                    },
                    'angle_values': {
                        'neck': float(row['Neck Angle']),
                        'waist': float(row['Waist Angle']),
                        'left_upper_arm': float(row['Left Upper Arm Angle']),
                        'right_upper_arm': float(row['Right Upper Arm Angle']),
                        'left_lower_arm': float(row['Left Lower Arm Angle']),
                        'right_lower_arm': float(row['Right Lower Arm Angle']),
                        'left_leg': float(row['Left Leg Angle']),
                        'right_leg': float(row['Right Leg Angle'])
                    }
                }
                single_frame_results.append(result)
                
            except Exception as e:
                print(f"Warning: Could not process frame {row.get('Frame', 'unknown')}: {e}")
                continue
    
    # Create summary from single frame results
    single_frame_summary = summarize_results(single_frame_results) if single_frame_results else None
    
    # Then, add temporal predictions if possible and available
    temporal_analysis = None
    if len(rows) >= 60 and resources.get('has_temporal_model', False):
        temporal_processor = resources.get('temporal_processor')
        if temporal_processor:
            temporal_analysis = temporal_processor.analyze_temporal_predictions(rows, resources)
    else:
        if len(rows) < 60:
            print(f"Video too short for temporal analysis ({len(rows)} < 60 frames)")
        if not resources.get('has_temporal_model', False):
            print("Temporal model not available, using single frame analysis only")
    
    # Determine primary results
    if temporal_analysis:
        # Use temporal results as primary
        avg_reba_score = temporal_analysis['avg_temporal_reba']
        prediction_method = "temporal"
        
        # Compare temporal vs single frame
        if single_frame_summary:
            single_avg = single_frame_summary['avg_reba_score']
            improvement = avg_reba_score - single_avg
            temporal_analysis['comparison_with_single_frame'] = {
                'single_frame_average': single_avg,
                'temporal_average': avg_reba_score,
                'improvement': improvement
            }
            print(f"Temporal vs Single Frame comparison:")
            print(f"   Single frame average: {single_avg:.3f}")
            print(f"   Temporal average: {avg_reba_score:.3f}")
            print(f"   Temporal improvement: {improvement:+.3f}")
    else:
        # Use single frame results as primary
        avg_reba_score = single_frame_summary['avg_reba_score'] if single_frame_summary else 0.0
        prediction_method = "single_frame"
    
    # Create enhanced results
    enhanced_results = {
        'avg_reba_score': avg_reba_score,
        'risk_level': get_risk_level(avg_reba_score),
        'prediction_method': prediction_method,
        'single_frame_results': single_frame_results,
        'single_frame_summary': single_frame_summary,
        'temporal_analysis': temporal_analysis
    }
    
    return enhanced_results


def enhance_video_segment_processing(all_rows, resources, processed_count):
    """
    Drop-in replacement for the segment processing results creation
    Add this to your process_video_segment function
    """
    from app.services.ergonomic_model import generate_feedback
    
    # Check if we have enough data
    if len(all_rows) < 3:
        print(f"Warning: Insufficient valid frames ({processed_count})")
        if processed_count == 0:
            return None
    
    # Create enhanced results with temporal analysis
    enhanced_results = create_enhanced_frame_results(all_rows, resources)
    
    # Add segment-specific info
    enhanced_results['processed_frames'] = processed_count
    
    # Calculate component scores for feedback
    if enhanced_results['single_frame_results']:
        component_scores = {}
        for component in ['trunk', 'neck', 'upper_arm', 'lower_arm']:
            scores = [r['component_scores'][component] for r in enhanced_results['single_frame_results'] 
                     if component in r['component_scores']]
            if scores:
                component_scores[component] = np.mean(scores)
        
        enhanced_results['avg_component_scores'] = component_scores
        
        # Generate feedback using component scores
        avg_component_scores = {
            'trunk_score': component_scores.get('trunk'),
            'neck_score': component_scores.get('neck'),
            'upper_arm_score': component_scores.get('upper_arm'),
            'lower_arm_score': component_scores.get('lower_arm'),
        }
        enhanced_results['feedback'] = generate_feedback(avg_component_scores, enhanced_results['avg_reba_score'])
    
    return enhanced_results


