import os
import shutil
import threading
import time
from datetime import datetime

# Track scheduled cleanups to avoid duplicates
_scheduled_cleanups = set()
_cleanup_lock = threading.Lock()

def schedule_cleanup(path, delay_seconds=60):
    """
    Schedule a file or directory for cleanup after a specified delay
    
    Args:
        path: Path to file or directory to clean up
        delay_seconds: Delay in seconds before cleaning up
    """
    with _cleanup_lock:
        if path in _scheduled_cleanups:
            return  # Already scheduled
        _scheduled_cleanups.add(path)
    
    def _delayed_cleanup():
        try:
            time.sleep(delay_seconds)
            
            if not os.path.exists(path):
                return
                
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleaned up directory: {path}")
            else:
                os.remove(path)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Cleaned up file: {path}")
                
        except Exception as e:
            print(f"Error during cleanup of {path}: {e}")
        finally:
            with _cleanup_lock:
                if path in _scheduled_cleanups:
                    _scheduled_cleanups.remove(path)
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=_delayed_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scheduled cleanup for {path} in {delay_seconds} seconds")
    return cleanup_thread

def cleanup_old_files(directory, max_age_days=1):
    """
    Clean up files in a directory that are older than max_age_days
    
    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
    """
    if not os.path.exists(directory):
        return
        
    current_time = time.time()
    max_age_seconds = max_age_days * 86400
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Skip if it's scheduled for cleanup
        if item_path in _scheduled_cleanups:
            continue
            
        # Get item age
        item_age = current_time - os.path.getmtime(item_path)
        
        # Remove if older than max age
        if item_age > max_age_seconds:
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Removed old item: {item_path}")
            except Exception as e:
                print(f"Error removing {item_path}: {e}")

def setup_periodic_cleanup():
    """Set up periodic cleanup of temporary directories"""
    def _periodic_cleanup():
        while True:
            try:
                # Clean up temp jobs directory
                cleanup_old_files("temp_jobs", max_age_days=1)
                
                # Clean up output images directory
                cleanup_old_files("output_images", max_age_days=1)
                
            except Exception as e:
                print(f"Error during periodic cleanup: {e}")
                
            # Run every hour
            time.sleep(3600)
    
    # Start periodic cleanup thread
    cleanup_thread = threading.Thread(target=_periodic_cleanup)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    return cleanup_thread
