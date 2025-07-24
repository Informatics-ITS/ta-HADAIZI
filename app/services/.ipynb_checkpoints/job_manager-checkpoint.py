import uuid
import time
import json
import os
from app.utils.file_cleanup import schedule_cleanup

# In-memory storage for jobs
jobs = {}

# Job expiration time (24 hours)
JOB_EXPIRE_SECONDS = 86400

def generate_random_id():
    """Generate a random UUID for job identification"""
    return str(uuid.uuid4())

def create_job():
    """Create a new job and store its metadata"""
    job_id = generate_random_id()
    jobs[job_id] = {
        "status": "processing",
        "result": None,
        "created_at": time.time()
    }
    return job_id

def update_job(job_id, result):
    """Update a job with its results"""
    if job_id in jobs:
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        jobs[job_id]["finished_at"] = time.time()
        jobs[job_id]["expire_at"] = time.time() + JOB_EXPIRE_SECONDS
        
        # Schedule cleanup of job folder
        job_folder = os.path.join("temp_jobs", job_id)
        if os.path.exists(job_folder):
            schedule_cleanup(job_folder, delay_seconds=600)  # 10 minutes
        
        # Save result to a JSON file for persistence (optional)
        try:
            # Convert NumPy types to Python native types
            def convert_numpy(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy(item) for item in item)
                elif hasattr(obj, 'tolist'):  # For numpy arrays
                    return obj.tolist()
                elif hasattr(obj, 'item'):  # For numpy scalars (int64, float64, etc.)
                    return obj.item()
                else:
                    return obj
            
            serializable_result = convert_numpy(result)
            
            result_path = os.path.join("temp_jobs", f"{job_id}_result.json")
            with open(result_path, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            schedule_cleanup(result_path, delay_seconds=JOB_EXPIRE_SECONDS)
        except Exception as e:
            print(f"Warning: Could not save job result to file: {e}")

def get_job(job_id):
    """Get job status and result"""
    job = jobs.get(job_id)
    
    # If job not in memory, try to load from file
    if job is None:
        try:
            result_path = os.path.join("temp_jobs", f"{job_id}_result.json")
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    result = json.load(f)
                # Create a minimal job entry
                return {
                    "status": "done",
                    "result": result
                }
        except Exception:
            pass
        return None
    
    # Check if job has expired
    if "expire_at" in job and time.time() > job["expire_at"]:
        del jobs[job_id]
        return None
    
    return job

def cleanup_expired_jobs():
    """Remove expired jobs from memory"""
    current_time = time.time()
    expired_jobs = []
    
    for job_id, job_data in jobs.items():
        if "expire_at" in job_data and current_time > job_data["expire_at"]:
            expired_jobs.append(job_id)
    
    for job_id in expired_jobs:
        del jobs[job_id]
        print(f"Cleaned up expired job: {job_id}")
