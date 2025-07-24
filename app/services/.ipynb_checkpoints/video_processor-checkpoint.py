import cv2
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import threading
import time
import tensorflow as tf

from app.services.pose_estimation import init_movenet, init_crop_region, get_joint_angles, create_row_dict
from app.services.ergonomic_model import get_model_resources, engineer_features_for_single_image, predict_single_image
from app.services.ergonomic_model import get_risk_level, generate_feedback
from app.services.job_manager import update_job
from app.utils.summarize_results import summarize_results

# Constants
FRAME_INTERVAL = 3  # Process every Nth frame
SEGMENT_DURATION_MINUTES = 5  # Default segment size in minutes
BATCH_SIZE = 32  # Process frames in batches for efficiency


def make_json_serializable(obj):
    """
    Recursively convert any NumPy types to native Python types to make them JSON serializable.
    """
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    elif hasattr(obj, 'tolist'):  # For numpy arrays
        return obj.tolist()
    elif hasattr(obj, 'item'):  # For numpy scalars (int64, float64, etc.)
        return obj.item()
    else:
        return obj


def update_progress(progress_file, progress_value):
    """Update progress file with current progress value"""
    with open(progress_file, 'w') as f:
        f.write(f"{progress_value:.1f}")


def process_frame(frame, movenet, input_size, frame_count, segment_index, resources):
    """Process a single video frame"""
    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Initialize crop region
    crop_region = init_crop_region(frame_rgb.shape[0], frame_rgb.shape[1])
    
    # Prepare input for MoveNet
    input_image = tf.expand_dims(frame_rgb, axis=0)
    input_image = tf.image.crop_and_resize(
        input_image,
        boxes=[[
            crop_region['y_min'],
            crop_region['x_min'],
            crop_region['y_max'],
            crop_region['x_max']
        ]],
        box_indices=[0],
        crop_size=[input_size, input_size]
    )
    
    # Run pose detection
    keypoints_with_scores = movenet(input_image)
    
    # Calculate joint angles
    angles = get_joint_angles(keypoints_with_scores)
    
    if angles is not None:
        # Create row dictionary
        row = create_row_dict(angles, f"segment_{segment_index+1}", frame_count)
        
        # Return the processed data
        return {
            'row': row,
            'keypoints': keypoints_with_scores,
            'crop_region': crop_region,
            'frame_rgb': frame_rgb,
            'angles': angles
        }
    
    return None


def process_frame_batch(frames_batch, movenet, input_size, start_frame_idx, segment_index, resources):
    """Process a batch of frames efficiently"""
    results = []
    
    for i, frame in enumerate(frames_batch):
        frame_idx = start_frame_idx + i
        
        try:
            frame_data = process_frame(frame, movenet, input_size, frame_idx, segment_index, resources)
            if frame_data:
                results.append(frame_data['row'])
        except Exception as e:
            print(f"Warning: Error processing frame {frame_idx}: {e}")
            continue
    
    return results


def save_debug_frame(frame_data, frame_count, resources, frames_folder):
    """Save a visualization of the processed frame for debugging"""
    try:
        # Create frame visualization
        row = frame_data['row']
        features, component_scores = engineer_features_for_single_image(row, resources)
        reba_score = predict_single_image(features, resources)
        component_scores['reba_score'] = reba_score
        
        # Generate visualization (import locally to avoid circular import)
        from app.services.image_visualizer import generate_pose_visualization
        
        keypoints_adjusted = frame_data['keypoints'].copy()
        crop_region = frame_data['crop_region']
        frame_rgb = frame_data['frame_rgb']
        
        # Adjust keypoints for crop region
        for idx in range(17):
            keypoints_adjusted[0, 0, idx, 0] = (
                crop_region['y_min'] * frame_rgb.shape[0] +
                crop_region['height'] * frame_rgb.shape[0] *
                keypoints_adjusted[0, 0, idx, 0]) / frame_rgb.shape[0]
            keypoints_adjusted[0, 0, idx, 1] = (
                crop_region['x_min'] * frame_rgb.shape[1] +
                crop_region['width'] * frame_rgb.shape[1] *
                keypoints_adjusted[0, 0, idx, 1]) / frame_rgb.shape[1]
        
        # Pass angle values to visualization for imputation indicators
        angle_values = frame_data['angles']
        
        visualization = generate_pose_visualization(
            frame_rgb, keypoints_adjusted, component_scores, 
            angle_values=angle_values
        )
        
        # Save visualization
        debug_path = os.path.join(frames_folder, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(debug_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        
        return debug_path
        
    except Exception as e:
        print(f"Warning: Could not save debug frame {frame_count}: {e}")
        return None


def create_frame_results(rows, resources):
    """Create analysis results for each frame using updated pipeline"""
    results = []
    
    # Process in smaller batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(rows), batch_size):
        batch_rows = rows[i:i+batch_size]
        df_batch = pd.DataFrame(batch_rows)
        
        for _, row in df_batch.iterrows():
            row_dict = row.to_dict()
            
            try:
                # Use the updated feature engineering pipeline
                features, component_scores = engineer_features_for_single_image(row_dict, resources)
                # Make prediction using the updated model
                reba_score = predict_single_image(features, resources)
                
                # Create result with updated structure
                result = {
                    'frame': int(row['Frame']),
                    'reba_score': float(reba_score),
                    'component_scores': {
                        'trunk': int(component_scores['trunk_score']),
                        'neck': int(component_scores['neck_score']),
                        'upper_arm': int(component_scores['upper_arm_score']),
                        'lower_arm': int(component_scores['lower_arm_score']),
                        # Note: leg_score excluded as per expert recommendation
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
                results.append(result)
                
            except Exception as e:
                print(f"Warning: Could not process frame {row.get('Frame', 'unknown')}: {e}")
                continue
    
    return results


def get_video_metadata(cap):
    """Extract metadata from video capture object"""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    return {
        'total_frames': total_frames,
        'fps': fps,
        'width': width,
        'height': height,
        'duration_seconds': duration_seconds
    }


def process_video_segment(cap, movenet, input_size, resources, start_frame, end_frame, total_frames, 
                         job_folder, segment_index, progress_file):
    """Process a segment of video frames and return the analysis results with batch optimization"""
    # Create frames folder for debug images (optional)
    frames_folder = os.path.join(job_folder, f"segment_{segment_index+1}_frames")
    os.makedirs(frames_folder, exist_ok=True)
    
    # Initialize tracking variables
    frame_count = start_frame
    processed_count = 0
    all_rows = []
    
    print(f"Processing segment {segment_index+1}: frames {start_frame}-{end_frame}")
    
    # Process frames in batches for efficiency
    while frame_count < end_frame:
        # Collect batch of frames
        batch_frames = []
        batch_frame_indices = []
        
        for _ in range(BATCH_SIZE):
            if frame_count >= end_frame:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process every FRAME_INTERVAL frames
            if (frame_count - start_frame) % FRAME_INTERVAL == 0:
                batch_frames.append(frame)
                batch_frame_indices.append(frame_count)
            
            frame_count += 1
        
        if not batch_frames:
            break
        
        # Process batch
        try:
            batch_results = process_frame_batch(
                batch_frames, movenet, input_size, 
                batch_frame_indices[0], segment_index, resources
            )
            
            if batch_results:
                all_rows.extend(batch_results)
                processed_count += len(batch_results)
                
                # Save debug frame for first batch only
                if len(all_rows) <= BATCH_SIZE and batch_results:
                    # Create frame data for visualization
                    frame_data = {
                        'row': batch_results[0],
                        'keypoints': None,  # We'd need to store this from process_frame
                        'crop_region': None,
                        'frame_rgb': None,
                        'angles': None
                    }
                    # Skip debug frame saving for batch processing
            
        except Exception as e:
            print(f"Warning: Batch processing error: {e}")
            continue
        
        # Update progress more frequently
        if frame_count % (FRAME_INTERVAL * 10) == 0:  # Every 30 frames
            overall_progress = min(100.0, 100.0 * (frame_count - start_frame) / (end_frame - start_frame))
            segment_weight = 1.0 / max(1, total_frames / (end_frame - start_frame))
            global_progress = min(100.0, 100.0 * (segment_index * segment_weight + 
                                 overall_progress / 100.0 * segment_weight))
            
            update_progress(progress_file, global_progress)
            print(f"  Progress: {global_progress:.1f}% ({processed_count} frames)")
    
    print(f"Segment {segment_index+1} processing complete: {processed_count} frames analyzed")
    
    # Check if we have enough data
    if len(all_rows) < 3:
        print(f"Warning: Insufficient valid frames in segment {segment_index+1} ({processed_count})")
        if processed_count == 0:
            return None
    
    # Create results for each frame using updated pipeline
    results = create_frame_results(all_rows, resources)
    
    # Summarize segment results
    summary = summarize_results(results)
    
    # Add segment-specific info
    summary['processed_frames'] = processed_count
    
    # Generate feedback using new Indonesian feedback system
    avg_component_scores = {
        'trunk_score': summary['avg_component_scores']['trunk'],
        'neck_score': summary['avg_component_scores']['neck'],
        'upper_arm_score': summary['avg_component_scores']['upper_arm'],
        'lower_arm_score': summary['avg_component_scores']['lower_arm'],
        # Note: leg_score excluded as per expert recommendation
    }
    
    summary['feedback'] = generate_feedback(avg_component_scores, summary['avg_reba_score'])
    
    return summary


def process_video(job_folder, job_id, video_path, segment_duration_minutes=None):
    """
    Process a video file for ergonomic analysis with optimizations, optionally dividing it into segments
    
    Args:
        job_folder: Folder containing job files
        job_id: Unique job identifier
        video_path: Path to video file
        segment_duration_minutes: Duration of each segment in minutes (None for default behavior)
    """
    try:
        print(f"Starting video processing job {job_id}")
        
        # Initialize MoveNet and resources
        movenet, input_size = init_movenet()
        if movenet is None:
            raise ValueError("Could not initialize MoveNet model")
            
        resources = get_model_resources()
        if resources is None:
            raise ValueError("Could not load model resources")
            
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # Get video metadata
        metadata = get_video_metadata(cap)
        total_frames = metadata['total_frames']
        fps = metadata['fps']
        duration_seconds = metadata['duration_seconds']
        
        print(f"Video info: {metadata['width']}x{metadata['height']}, {fps:.2f} fps, "
              f"{duration_seconds:.2f} seconds, {total_frames} frames")
        print(f"Processing every {FRAME_INTERVAL} frames")
        print(f"Estimated processing frames: {total_frames // FRAME_INTERVAL}")
        
        # Handle None segment_duration_minutes properly
        if segment_duration_minutes is None:
            segment_duration_minutes = SEGMENT_DURATION_MINUTES  # Use default
        
        # Determine if video should be segmented
        use_segments = (segment_duration_minutes > 0 and 
                       duration_seconds > segment_duration_minutes * 60)
        
        if use_segments:
            segment_frames = int(segment_duration_minutes * 60 * fps)
            num_segments = (total_frames + segment_frames - 1) // segment_frames  # Ceiling division
            print(f"Video will be processed in {num_segments} segments of {segment_duration_minutes} minutes each")
        else:
            num_segments = 1
            segment_frames = total_frames
            print("Processing entire video as a single segment")
        
        # Create progress tracking file
        progress_file = os.path.join(job_folder, "progress.txt")
        update_progress(progress_file, 0.0)
        
        # Process video in segments
        all_segment_results = []
        
        for segment_index in range(num_segments):
            start_frame = segment_index * segment_frames
            end_frame = min((segment_index + 1) * segment_frames, total_frames)
            
            # Skip to segment start
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Process this segment
            segment_result = process_video_segment(
                cap, movenet, input_size, resources, 
                start_frame, end_frame, total_frames,
                job_folder, segment_index, progress_file
            )
            
            if segment_result:
                segment_result['segment_info'] = {
                    'segment_index': segment_index,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'start_time': start_frame / fps if fps > 0 else 0,
                    'end_time': end_frame / fps if fps > 0 else 0,
                }
                all_segment_results.append(segment_result)
        
        cap.release()
        
        # Create combined result with all segments
        video_name = os.path.basename(video_path)
        final_result = create_final_result(all_segment_results, video_name, metadata, 
                                          use_segments, segment_duration_minutes)
        
        # Ensure progress is 100%
        update_progress(progress_file, 100.0)
            
        # Make result JSON serializable
        final_result = make_json_serializable(final_result)
            
        # Update job with results
        update_job(job_id, final_result)
        
        print(f"Video analysis completed for {video_name}")
        return final_result
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update job with error
        error_result = {
            "error": str(e),
            "status": "failed"
        }
        update_job(job_id, error_result)
        
        return error_result
    finally:
        # Ensure progress is 100% when finished
        try:
            progress_file = os.path.join(job_folder, "progress.txt")
            update_progress(progress_file, 100.0)
        except:
            pass


def create_final_result(all_segment_results, video_name, metadata, use_segments, segment_duration_minutes):
    """Create the final result object combining all segment results"""
    if not all_segment_results:
        return None
        
    if use_segments and len(all_segment_results) > 1:
        # Multiple segments - create a parent result
        final_result = {
            'video_metadata': {
                'filename': video_name,
                'duration_seconds': metadata['duration_seconds'],
                'total_frames': metadata['total_frames'],
                'fps': float(metadata['fps']),
                'segments_count': len(all_segment_results),
                'segment_duration_minutes': segment_duration_minutes
            },
            'segments': all_segment_results
        }
        
        # Calculate overall average REBA score
        reba_scores = [s['avg_reba_score'] for s in all_segment_results]
        final_result['overall_avg_reba_score'] = float(np.mean(reba_scores))
        final_result['overall_risk_level'] = get_risk_level(final_result['overall_avg_reba_score'])
        
        # Find highest risk segment
        highest_risk_index = np.argmax(reba_scores)
        final_result['highest_risk_segment'] = {
            'segment_index': highest_risk_index,
            'segment_reba_score': reba_scores[highest_risk_index],
            'segment_time': all_segment_results[highest_risk_index]['segment_info']
        }
        
        # Combined recommendations (now in Indonesian)
        all_recommendations = set()
        for segment in all_segment_results:
            if 'recommendations' in segment:
                all_recommendations.update(segment['recommendations'])
        final_result['overall_recommendations'] = list(all_recommendations)
    else:
        # Single segment - use its result as the final result
        final_result = all_segment_results[0]
        
        # Add video metadata for single segment
        final_result['video_metadata'] = {
            'filename': video_name,
            'duration_seconds': metadata['duration_seconds'],
            'total_frames': metadata['total_frames'],
            'fps': float(metadata['fps']),
            'segments_count': 1,
            'segment_duration_minutes': 0  # Single segment
        }
    
    return final_result