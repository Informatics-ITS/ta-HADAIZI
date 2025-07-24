from flask import Blueprint, request, jsonify, send_from_directory, render_template
import os
import numpy as np
from datetime import datetime, timedelta
from app.services.pose_estimation import process_pose_from_bytes
from app.services.video_processor import process_video
from app.services.job_manager import create_job, get_job, update_job
from app.utils.file_cleanup import schedule_cleanup

ergonomic_bp = Blueprint('ergonomic', __name__)

@ergonomic_bp.route('/predict/image', methods=['POST'])
def predict_image():
    """Endpoint for analyzing a single image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()
    
    try:
        # Process image and get predictions
        result = process_pose_from_bytes(image_bytes)
        
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ergonomic_bp.route('/output_images/<path:filename>')
def serve_output_image(filename):
    """Serve output images"""
    # Extract date part from filename (expected format: YYYY-MM-DD/something.png)
    parts = filename.split('/')
    if len(parts) > 1:
        date_dir = parts[0]
        file_name = parts[1]
        directory = os.path.join(os.getcwd(), 'output_images', date_dir)
    else:
        directory = os.path.join(os.getcwd(), 'output_images')
        file_name = filename
        
    return send_from_directory(directory, file_name, mimetype='image/png')

@ergonomic_bp.route("/predict/video", methods=["POST"])
def predict_video():
    """Endpoint for analyzing a video file"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    
    # Get segment duration parameter (if provided)
    segment_duration_minutes = request.form.get('segment_duration_minutes', None)
    if segment_duration_minutes:
        try:
            segment_duration_minutes = int(segment_duration_minutes)
            print(f"Using custom segment duration: {segment_duration_minutes} minutes")
        except ValueError:
            segment_duration_minutes = None
            print("Invalid segment duration parameter, using default")
    
    # Create a job ID for async processing
    job_id = create_job()
    
    # Create temporary directory for job files
    job_folder = os.path.join("temp_jobs", job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    # Save the uploaded video
    video_path = os.path.join(job_folder, "video.mp4")
    file.save(video_path)
    
    # Start processing in background
    import threading
    
    # Pass segment_duration_minutes to the process_video function
    threading.Thread(target=process_video, 
                     args=(job_folder, job_id, video_path, segment_duration_minutes)).start()
    
    return jsonify({"job_id": job_id})

@ergonomic_bp.route("/predict/video/result", methods=["GET"])
def get_video_result():
    """Get the result of an asynchronous video job"""
    job_id = request.args.get("job_id")
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found or expired"}), 404
    
    # Convert any NumPy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    serializable_job = convert_numpy(job)
    return jsonify(serializable_job)

@ergonomic_bp.route('/predict/video/status', methods=['GET'])
def get_video_status():
    """Get processing status for a video job"""
    job_id = request.args.get("job_id")
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
        
    # Check if progress file exists
    progress_file = os.path.join("temp_jobs", job_id, "progress.txt")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = float(f.read().strip())
            return jsonify({"job_id": job_id, "progress": progress})
        except:
            pass
    
    # Fall back to basic status from job
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found or expired"}), 404
    
    status = job.get("status", "unknown")
    progress = 100.0 if status == "done" else 0.0
    
    return jsonify({"job_id": job_id, "status": status, "progress": progress})