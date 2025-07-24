#!/usr/bin/env python3
"""
Enhanced test script for ergonomic assessment model
Tests ALL images AND videos using your complete pipeline
INCLUDING YOUR PRODUCTION VIDEO PROCESSING SYSTEM
Outputs results in JSON format with angle values
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import cv2
from pathlib import Path
import traceback
from datetime import datetime
import uuid
import tempfile
import shutil
import json
import zipfile

# Add visualization imports
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add current directory to path to import app modules
sys.path.append(os.getcwd())

# Import ALL your app modules
try:
    # Import from app.services.pose_estimation
    from app.services.pose_estimation import (
        init_movenet, get_joint_angles, create_row_dict, init_crop_region,
        run_inference, should_flip_image, crop_and_resize, KEYPOINT_DICT,
        calculate_angle, process_pose_from_bytes, AngleSmoother,
        get_keypoint_if_valid, calculate_angle_with_fallback
    )
    
    # Import from app.services.ergonomic_model
    from app.services.ergonomic_model import (
        get_model_resources, engineer_features_for_single_image, predict_single_image,
        get_risk_level, generate_feedback, get_action_level, predict_video,
        prepare_sequences, calculate_component_scores_from_angles
    )
    
    # Import from app.services.image_visualizer
    from app.services.image_visualizer import generate_pose_visualization
    
    # ADDED: Import your video processing pipeline
    from app.services.video_processor import process_video
    
    LOCAL_MODULES_AVAILABLE = True
    print("ALL app modules imported successfully!")
    
except ImportError as e:
    print(f"Failed to import app modules: {e}")
    print("Please ensure you're running from the root directory with app/ folder")
    LOCAL_MODULES_AVAILABLE = False

# Constants
KEYPOINT_THRESHOLD = 0.3
SEQUENCE_LENGTH = 60
TEST_DIRECTORIES = ['test']
OUTPUT_VIZ_DIR = "test_visualizationsv2"
OUTPUT_RESULTS_JSON = "test_results.json"
OUTPUT_VIDEO_RESULTS_JSON = "video_test_results.json"
TEMP_JOBS_DIR = "temp_test_jobs"

# Global tracking for missing keypoints
missing_keypoints_log = []

def log_missing_keypoints(filename, frame_number, missing_angles, debug_info):
    """Log missing keypoints information to global list"""
    log_entry = {
        'filename': filename,
        'frame_number': frame_number,
        'timestamp': datetime.now().isoformat(),
        'missing_angles': missing_angles,
        'keypoint_details': {}
    }
    
    # Parse debug info to extract keypoint confidences
    for info in debug_info:
        if 'NECK missing keypoints:' in info:
            neck_data = info.replace('NECK missing keypoints: ', '')
            neck_keypoints = {}
            for kp_conf in neck_data.split(', '):
                if ':' in kp_conf:
                    kp, conf = kp_conf.split(':')
                    neck_keypoints[kp] = float(conf)
            log_entry['keypoint_details']['neck'] = neck_keypoints
        elif 'WAIST missing keypoints:' in info:
            waist_data = info.replace('WAIST missing keypoints: ', '')
            waist_keypoints = {}
            for kp_conf in waist_data.split(', '):
                if ':' in kp_conf:
                    kp, conf = kp_conf.split(':')
                    waist_keypoints[kp] = float(conf)
            log_entry['keypoint_details']['waist'] = waist_keypoints
    
    missing_keypoints_log.append(log_entry)

def save_missing_keypoints_json():
    """Save missing keypoints log to JSON file"""
    if missing_keypoints_log:
        json_path = os.path.join(OUTPUT_VIZ_DIR, "missing_keypoints_log.json")
        with open(json_path, 'w') as f:
            json.dump(missing_keypoints_log, f, indent=2)
        print(f"Missing keypoints log saved: {json_path}")
        return json_path
    return None

def create_results_zip():
    """Create a zip file containing all visualizations and JSON files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"test_results_{timestamp}.zip"
    zip_path = os.path.join(OUTPUT_VIZ_DIR, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all visualization files
            viz_files = glob.glob(os.path.join(OUTPUT_VIZ_DIR, "*.png"))
            for viz_file in viz_files:
                arcname = os.path.basename(viz_file)
                zipf.write(viz_file, arcname)
            
            # Add JSON files
            json_files = [OUTPUT_RESULTS_JSON, OUTPUT_VIDEO_RESULTS_JSON]
            for json_file in json_files:
                if os.path.exists(json_file):
                    zipf.write(json_file, os.path.basename(json_file))
            
            # Add missing keypoints log if it exists
            missing_kp_file = os.path.join(OUTPUT_VIZ_DIR, "missing_keypoints_log.json")
            if os.path.exists(missing_kp_file):
                zipf.write(missing_kp_file, "missing_keypoints_log.json")
            
            # Add summary file
            summary_data = {
                'test_timestamp': datetime.now().isoformat(),
                'total_visualizations': len(viz_files),
                'total_missing_keypoints_entries': len(missing_keypoints_log),
                'visualization_files': [os.path.basename(f) for f in viz_files],
                'json_files': [os.path.basename(f) for f in json_files if os.path.exists(f)]
            }
            
            summary_json = json.dumps(summary_data, indent=2)
            zipf.writestr("test_summary.json", summary_json)
        
        print(f"Results zip created: {zip_path}")
        return zip_path
        
    except Exception as e:
        print(f"Error creating zip file: {e}")
        return None

def save_visualization_with_app_modules(processed_image, keypoints_with_scores, component_scores, 
                                       reba_score, image_filename, angle_values=None, 
                                       original_image=None, flip_applied=False):
    """Save visualization using your app modules"""
    if not LOCAL_MODULES_AVAILABLE:
        print("App modules not available for visualization")
        return None
        
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    
    # Add REBA score to component scores for your visualizer
    component_scores_with_reba = component_scores.copy()
    component_scores_with_reba['reba_score'] = reba_score
    
    # Use your app's generate_pose_visualization function
    try:
        vis_image = generate_pose_visualization(
            processed_image, keypoints_with_scores, component_scores_with_reba, 
            original_image, flip_applied
        )
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(image_filename)[0]
        viz_filename = f"{base_name}_reba_{reba_score:.1f}_{timestamp}.png"
        viz_path = os.path.join(OUTPUT_VIZ_DIR, viz_filename)
        
        # Convert RGB to BGR for cv2 saving
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(viz_path, vis_image_bgr)
        
        return viz_path
        
    except Exception as e:
        print(f"Error generating visualization: {e}")
        traceback.print_exc()
        return None

def find_test_files():
    """Find all image and video files in test directories"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    test_files = {'images': [], 'videos': []}
    
    # Check current directory
    for ext in image_extensions:
        test_files['images'].extend(glob.glob(f"*{ext}"))
        test_files['images'].extend(glob.glob(f"*{ext.upper()}"))
    
    # Check test directories
    for test_dir in TEST_DIRECTORIES:
        if os.path.exists(test_dir):
            print(f"Checking directory: {test_dir}")
            
            for ext in image_extensions + video_extensions:
                pattern = os.path.join(test_dir, f"*{ext}")
                files = glob.glob(pattern)
                if ext in image_extensions:
                    test_files['images'].extend(files)
                else:
                    test_files['videos'].extend(files)
                
                pattern = os.path.join(test_dir, f"*{ext.upper()}")
                files = glob.glob(pattern)
                if ext in image_extensions:
                    test_files['images'].extend(files)
                else:
                    test_files['videos'].extend(files)
    
    # Remove duplicates and SORT FILES IN NATURAL ORDER
    test_files['images'] = list(set(test_files['images']))
    test_files['videos'] = list(set(test_files['videos']))
    
    # Natural sort function for proper numerical ordering
    import re
    def natural_sort_key(path):
        filename = os.path.basename(path)
        # Split filename into text and numbers for proper sorting
        parts = re.split(r'(\d+)', filename)
        # Convert numeric parts to integers for proper sorting
        return [int(part) if part.isdigit() else part.lower() for part in parts]
    
    # Sort both lists naturally
    test_files['images'].sort(key=natural_sort_key)
    test_files['videos'].sort(key=natural_sort_key)
    
    print(f"Found {len(test_files['images'])} images and {len(test_files['videos'])} videos")
    
    # Print the order for verification
    if test_files['images']:
        print("Image processing order:")
        for i, img_path in enumerate(test_files['images'], 1):
            print(f"  {i:2d}. {os.path.basename(img_path)}")
    
    if test_files['videos']:
        print("Video processing order:")
        for i, vid_path in enumerate(test_files['videos'], 1):
            print(f"  {i:2d}. {os.path.basename(vid_path)}")
    
    return test_files

def test_image_processing_with_app_modules(image_path, test_number, total_tests):
    """Test image processing using ALL your app modules"""
    print(f"\nTesting image {test_number}/{total_tests}: {os.path.basename(image_path)}")
    
    if not LOCAL_MODULES_AVAILABLE:
        print("App modules not available")
        return None, False
    
    result_data = {
        'image_name': os.path.basename(image_path),
        'image_path': image_path,
        'test_number': test_number,
        'success': False,
        'reba_score': None,
        'risk_level': None,
        'neck_score': None,
        'trunk_score': None,
        'upper_arm_score': None,
        'lower_arm_score': None,
        'wrist_score': None,
        'leg_score': None,
        'load_score': None,
        'coupling_score': None,
        'activity_score': None,
        'neck_angle': None,
        'waist_angle': None,
        'left_upper_arm_angle': None,
        'right_upper_arm_angle': None,
        'left_lower_arm_angle': None,
        'right_lower_arm_angle': None,
        'left_leg_angle': None,
        'right_leg_angle': None,
        'flip_applied': False,
        'visualization_path': None,
        'error_message': None,
        'processing_time': None,
        'missing_critical_keypoints': None,
        'keypoint_debug_info': None
    }
    
    start_time = datetime.now()
    
    try:
        # Read image as bytes (like your deployment)
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # Load the image for angle extraction - DO INFERENCE ONLY ONCE
        frame = cv2.imread(image_path)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Initialize and run inference for keypoints and angles
        movenet, input_size = init_movenet()
        crop_region = init_crop_region(frame_rgb.shape[0], frame_rgb.shape[1])
        keypoints_with_scores, processed_image, original_image, flip_required = run_inference(
            movenet, frame_rgb, crop_region, crop_size=[input_size, input_size]
        )
        
        # Extract keypoint confidences for debug logging
        keypoint_confidences = {}
        scores = keypoints_with_scores[0, 0, :, 2]
        for name, idx in KEYPOINT_DICT.items():
            keypoint_confidences[name] = float(scores[idx])
        
        # Try to get angles and capture debug info
        angles_result = None
        keypoint_debug_info = {
            'all_keypoint_confidences': keypoint_confidences,
            'missing_keypoints': {},
            'insufficient_keypoints': False
        }
        
        # Check critical keypoints for neck and waist
        critical_missing = {}
        
        # Check neck keypoints (ears + shoulders)
        neck_keypoints = ['left_ear', 'right_ear', 'left_shoulder', 'right_shoulder']
        neck_missing = []
        for kp in neck_keypoints:
            if keypoint_confidences[kp] < KEYPOINT_THRESHOLD:
                neck_missing.append(f"{kp}:{keypoint_confidences[kp]:.3f}")
        if neck_missing:
            critical_missing['neck'] = neck_missing
        
        # Check waist keypoints (shoulders + hips)
        waist_keypoints = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        waist_missing = []
        for kp in waist_keypoints:
            if keypoint_confidences[kp] < KEYPOINT_THRESHOLD:
                waist_missing.append(f"{kp}:{keypoint_confidences[kp]:.3f}")
        if waist_missing:
            critical_missing['waist'] = waist_missing
        
        # Try to get angles (this will print debug info once)
        try:
            angles_result = get_joint_angles(keypoints_with_scores)
        except Exception as angle_error:
            print(f"   Error calculating angles: {angle_error}")
            keypoint_debug_info['angle_calculation_error'] = str(angle_error)
        
        if angles_result is None:
            keypoint_debug_info['insufficient_keypoints'] = True
            keypoint_debug_info['missing_keypoints'] = critical_missing
            
            # Log this to global missing keypoints log
            log_missing_keypoints(
                os.path.basename(image_path),
                0,
                list(critical_missing.keys()),
                [f"{angle.upper()} missing keypoints: {', '.join(kps)}" for angle, kps in critical_missing.items()]
            )
        
        # Now try process_pose_from_bytes (but it will call get_joint_angles again - causing duplicate debug)
        try:
            result = process_pose_from_bytes(image_bytes, output_visualization=False)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            print(f"   Image processed successfully in {processing_time:.2f}s")
            print(f"   REBA Score: {result['reba_score']:.2f}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Component Scores: {result['component_scores']}")
            
            # Update result data
            result_data.update({
                'success': True,
                'reba_score': result['reba_score'],
                'risk_level': result['risk_level'],
                'processing_time': processing_time,
                'flip_applied': flip_required,
                'keypoint_debug_info': keypoint_debug_info
            })

            
            
            # Extract component scores
            component_scores = result['component_scores']
            result_data.update({
                'neck_score': component_scores.get('neck_score'),
                'trunk_score': component_scores.get('trunk_score'),
                'upper_arm_score': component_scores.get('upper_arm_score'),
                'lower_arm_score': component_scores.get('lower_arm_score'),
                'wrist_score': component_scores.get('wrist_score'),
                'leg_score': component_scores.get('leg_score'),
                'load_score': component_scores.get('load_score'),
                'coupling_score': component_scores.get('coupling_score'),
                'activity_score': component_scores.get('activity_score')
            })
            
            # Extract angle values from the result (since process_pose_from_bytes succeeded)
            if 'angle_values' in result:
                angle_values = result['angle_values']
                result_data.update({
                    'neck_angle': round(angle_values.get('neck'), 2) if angle_values.get('neck') is not None else None,
                    'waist_angle': round(angle_values.get('waist'), 2) if angle_values.get('waist') is not None else None,
                    'left_upper_arm_angle': round(angle_values.get('left_upper_arm'), 2) if angle_values.get('left_upper_arm') is not None else None,
                    'right_upper_arm_angle': round(angle_values.get('right_upper_arm'), 2) if angle_values.get('right_upper_arm') is not None else None,
                    'left_lower_arm_angle': round(angle_values.get('left_lower_arm'), 2) if angle_values.get('left_lower_arm') is not None else None,
                    'right_lower_arm_angle': round(angle_values.get('right_lower_arm'), 2) if angle_values.get('right_lower_arm') is not None else None,
                    'left_leg_angle': round(angle_values.get('left_leg'), 2) if angle_values.get('left_leg') is not None else None,
                    'right_leg_angle': round(angle_values.get('right_leg'), 2) if angle_values.get('right_leg') is not None else None
                })
            
            # Generate visualization
            print("   Generating visualization...")
            viz_path = save_visualization_with_app_modules(
                processed_image, keypoints_with_scores, result['component_scores'],
                result['reba_score'], os.path.basename(image_path), 
                result.get('angle_values'), 
                original_image, flip_required
            )
            
            if viz_path:
                result_data['visualization_path'] = viz_path
                print(f"   Visualization saved: {os.path.basename(viz_path)}")
            
            return result_data, True
            
        except Exception as process_error:
            # process_pose_from_bytes failed, but we still have debug info
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(process_error)
            
            result_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time': processing_time,
                'flip_applied': flip_required,
                'keypoint_debug_info': keypoint_debug_info
            })
            
            print(f"   Error processing image: {error_msg}")
            print(f"   Debug info: {keypoint_debug_info}")
            
            return result_data, False
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        result_data.update({
            'success': False,
            'error_message': error_msg,
            'processing_time': processing_time
        })
        print(f"   Error processing image: {error_msg}")
        return result_data, False

def test_video_processing_with_production_pipeline(video_path, test_number, total_tests, segment_duration=None):
    """Test video processing using your PRODUCTION video processing pipeline"""
    print(f"\nTesting video {test_number}/{total_tests}: {os.path.basename(video_path)}")
    
    if not LOCAL_MODULES_AVAILABLE:
        print("App modules not available")
        return None, False
    
    # Create temporary job folder
    job_id = f"test_video_{test_number}_{uuid.uuid4().hex[:8]}"
    job_folder = os.path.join(TEMP_JOBS_DIR, job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    result_data = {
        'video_name': os.path.basename(video_path),
        'video_path': video_path,
        'test_number': test_number,
        'job_id': job_id,
        'success': False,
        'avg_reba_score': None,
        'overall_risk_level': None,
        'segments_count': None,
        'segment_duration_minutes': segment_duration,
        'total_frames_processed': None,
        'video_duration_seconds': None,
        'processing_time': None,
        'error_message': None,
        'highest_risk_segment': None,
        'lowest_risk_segment': None,
        'risk_distribution': None
    }
    
    start_time = datetime.now()
    
    try:
        print("   Using your PRODUCTION video processing pipeline")
        print(f"   Job folder: {job_folder}")
        
        # Use your production video processing function
        result = process_video(
            job_folder=job_folder,
            job_id=job_id,
            video_path=video_path,
            segment_duration_minutes=segment_duration
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if result and 'error' not in result:
            print(f"   Video processed successfully in {processing_time:.2f}s")
            
            # Extract key metrics from result
            result_data.update({
                'success': True,
                'processing_time': processing_time
            })
            
            # Handle single segment vs multi-segment results
            if 'segments' in result:
                # Multi-segment video
                segments = result['segments']
                result_data.update({
                    'avg_reba_score': result.get('overall_avg_reba_score'),
                    'overall_risk_level': result.get('overall_risk_level'),
                    'segments_count': len(segments),
                    'video_duration_seconds': result['video_metadata']['duration_seconds'],
                    'total_frames_processed': sum(s.get('processed_frames', 0) for s in segments)
                })
                
                # Analyze segments
                segment_reba_scores = [s['avg_reba_score'] for s in segments]
                highest_idx = np.argmax(segment_reba_scores)
                lowest_idx = np.argmin(segment_reba_scores)
                
                result_data.update({
                    'highest_risk_segment': {
                        'index': highest_idx,
                        'reba_score': segment_reba_scores[highest_idx],
                        'start_time': segments[highest_idx]['segment_info']['start_time'],
                        'end_time': segments[highest_idx]['segment_info']['end_time']
                    },
                    'lowest_risk_segment': {
                        'index': lowest_idx,
                        'reba_score': segment_reba_scores[lowest_idx],
                        'start_time': segments[lowest_idx]['segment_info']['start_time'],
                        'end_time': segments[lowest_idx]['segment_info']['end_time']
                    }
                })
                
                print(f"   REBA Score: {result_data['avg_reba_score']:.2f}")
                print(f"   Risk Level: {result_data['overall_risk_level']}")
                print(f"   Segments: {len(segments)}")
                print(f"   Highest Risk: Segment {highest_idx+1} (REBA: {segment_reba_scores[highest_idx]:.2f})")
                print(f"   Lowest Risk: Segment {lowest_idx+1} (REBA: {segment_reba_scores[lowest_idx]:.2f})")
                
            else:
                # Single segment video
                result_data.update({
                    'avg_reba_score': result.get('avg_reba_score'),
                    'overall_risk_level': result.get('risk_level'),
                    'segments_count': 1,
                    'video_duration_seconds': result['video_metadata']['duration_seconds'],
                    'total_frames_processed': result.get('processed_frames', 0)
                })
                
                print(f"   REBA Score: {result_data['avg_reba_score']:.2f}")
                print(f"   Risk Level: {result_data['overall_risk_level']}")
                print(f"   Single segment processing")
            
            # Calculate risk distribution
            if 'segments' in result:
                risk_levels = [get_risk_level(s['avg_reba_score']) for s in result['segments']]
            else:
                risk_levels = [result_data['overall_risk_level']]
            
            risk_distribution = {}
            for risk in risk_levels:
                risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
            result_data['risk_distribution'] = risk_distribution
            
            return result_data, True
            
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            result_data.update({
                'success': False,
                'error_message': error_msg,
                'processing_time': processing_time
            })
            print(f"   Video processing failed: {error_msg}")
            return result_data, False
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        error_msg = str(e)
        result_data.update({
            'success': False,
            'error_message': error_msg,
            'processing_time': processing_time
        })
        print(f"   Error processing video: {error_msg}")
        traceback.print_exc()
        return result_data, False
    
    finally:
        # Clean up temporary job folder
        try:
            if os.path.exists(job_folder):
                shutil.rmtree(job_folder)
        except:
            pass

def save_results_to_json(results_list, filename):
    """Save all test results to JSON file"""
    if not results_list:
        print(f"No results to save to {filename}")
        return
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON serialization
    cleaned_results = []
    for result in results_list:
        cleaned_result = {}
        for key, value in result.items():
            cleaned_result[key] = convert_numpy_types(value)
        cleaned_results.append(cleaned_result)
    
    # Save to JSON with proper formatting
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(cleaned_results),
                "successful_tests": len([r for r in cleaned_results if r.get('success', False)])
            },
            "results": cleaned_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filename}")
    
    # Print summary statistics
    successful_tests = [r for r in cleaned_results if r.get('success', False)]
    if len(successful_tests) > 0:
        print(f"\nSummary Statistics for {filename}:")
        
        # Calculate REBA score statistics
        reba_scores = []
        for test in successful_tests:
            if 'reba_score' in test and test['reba_score'] is not None:
                reba_scores.append(test['reba_score'])
            elif 'avg_reba_score' in test and test['avg_reba_score'] is not None:
                reba_scores.append(test['avg_reba_score'])
        
        if reba_scores:
            print(f"Average REBA Score: {sum(reba_scores)/len(reba_scores):.2f}")
            print(f"Min REBA Score: {min(reba_scores):.2f}")
            print(f"Max REBA Score: {max(reba_scores):.2f}")
        
        # Calculate processing time statistics
        processing_times = [test['processing_time'] for test in successful_tests if test.get('processing_time') is not None]
        if processing_times:
            print(f"Average Processing Time: {sum(processing_times)/len(processing_times):.2f}s")

def main():
    """Main testing function using ALL your app modules - TESTING ALL IMAGES AND VIDEOS"""
    print("Starting Complete Ergonomic Assessment Model Test")
    print("TESTING ALL IMAGES + ALL VIDEOS WITH PRODUCTION PIPELINE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Output directory: {OUTPUT_VIZ_DIR}")
    print(f"Image results file: {OUTPUT_RESULTS_JSON}")
    print(f"Video results file: {OUTPUT_VIDEO_RESULTS_JSON}")
    
    # Check if app modules are available
    if not LOCAL_MODULES_AVAILABLE:
        print("\nApp modules not available!")
        print("Please ensure:")
        print("1. You're running from the root directory")
        print("2. The app/ folder exists with all modules")
        print("3. All dependencies are installed")
        return False
    
    # Create output directories
    os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
    os.makedirs(TEMP_JOBS_DIR, exist_ok=True)
    print(f"Output directories created")
    
    # Test your app modules initialization
    print(f"\nTesting your app modules initialization...")
    try:
        movenet, input_size = init_movenet()
        if movenet is None:
            print("Failed to initialize MoveNet")
            return False
        print(f"MoveNet initialized: input_size={input_size}")
        
        resources = get_model_resources()
        if resources is None:
            print("Failed to load model resources")
            return False
        print(f"Model resources loaded: {len(resources['model_features'])} features")
        
    except Exception as e:
        print(f"Error testing app modules: {e}")
        traceback.print_exc()
        return False
    
    # Find test files
    print(f"\nFinding test files...")
    test_files = find_test_files()
    
    if not test_files['images'] and not test_files['videos']:
        print("No test files found!")
        print("Please add some images or videos to the following directories:")
        for dir_name in TEST_DIRECTORIES:
            print(f"   - {dir_name}/")
        return False
    
    success_count = 0
    total_tests = 0
    
    # Test ALL IMAGES
    image_results = []
    if test_files['images']:
        print(f"\nTesting ALL IMAGES ({len(test_files['images'])} images)...")
        for i, image_path in enumerate(test_files['images'], 1):
            total_tests += 1
            result_data, success = test_image_processing_with_app_modules(image_path, i, len(test_files['images']))
            image_results.append(result_data)
            if success:
                success_count += 1
    
    # Test ALL VIDEOS using production pipeline
    video_results = []
    if test_files['videos']:
        print(f"\nTesting ALL VIDEOS with PRODUCTION PIPELINE ({len(test_files['videos'])} videos)...")
        
        # Ask user about segmentation for videos
        use_segmentation = False
        segment_duration = None
        
        if len(test_files['videos']) > 0:
            print(f"\nVideo Processing Options:")
            print(f"1. Process videos as single segments (faster)")
            print(f"2. Process videos with segmentation (more detailed analysis)")
            
            choice = input("Choose option (1 or 2) [default: 1]: ").strip()
            if choice == "2":
                use_segmentation = True
                try:
                    segment_duration = int(input("Enter segment duration in minutes [default: 5]: ").strip() or "5")
                except:
                    segment_duration = 5
                print(f"Using segmentation: {segment_duration} minute segments")
            else:
                print("Using single segment processing")
        
        for i, video_path in enumerate(test_files['videos'], 1):
            total_tests += 1
            result_data, success = test_video_processing_with_production_pipeline(
                video_path, i, len(test_files['videos']), 
                segment_duration if use_segmentation else None
            )
            video_results.append(result_data)
            if success:
                success_count += 1
    
    # Save results
    if image_results:
        save_results_to_json(image_results, OUTPUT_RESULTS_JSON)
    if video_results:
        save_results_to_json(video_results, OUTPUT_VIDEO_RESULTS_JSON)
    
    # Save missing keypoints JSON
    print(f"\nSaving missing keypoints log...")
    json_path = save_missing_keypoints_json()
    
    # Create zip file with all results
    print(f"\nCreating results zip file...")
    zip_path = create_results_zip()
    
    # Final Summary
    print(f"\nCOMPLETE TEST SUMMARY")
    print("=" * 80)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Images tested: {len(test_files['images'])}")
    print(f"Videos tested: {len(test_files['videos'])}")
    print(f"Total tests: {total_tests}")
    print(f"Successful tests: {success_count}/{total_tests}")
    print(f"Success rate: {(success_count/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    print(f"Visualizations saved in: {OUTPUT_VIZ_DIR}/")
    print(f"Image results: {OUTPUT_RESULTS_JSON}")
    print(f"Video results: {OUTPUT_VIDEO_RESULTS_JSON}")
    print(f"Missing keypoints entries: {len(missing_keypoints_log)}")
    if json_path:
        print(f"Missing keypoints log: {json_path}")
    if zip_path:
        print(f"Results zip file: {zip_path}")
    print(f"Used YOUR COMPLETE PRODUCTION PIPELINE!")
    
    # Clean up temp directory
    try:
        if os.path.exists(TEMP_JOBS_DIR):
            shutil.rmtree(TEMP_JOBS_DIR)
        print(f"Cleaned up temporary files")
    except:
        pass
    
    return success_count > 0

if __name__ == "__main__":
    try:
        success = main()
        exit_code = 0 if success else 1
        print(f"\nExiting with code: {exit_code}")
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)