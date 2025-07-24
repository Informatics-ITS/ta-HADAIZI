import sys
import os
import matplotlib
matplotlib.use('Agg')

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.utils.file_cleanup import setup_periodic_cleanup

# Create Flask application
app = create_app()

# Start background cleanup
cleanup_thread = setup_periodic_cleanup()

if __name__ == '__main__':
    # Run the application with standard Flask
    app.run(host='0.0.0.0', port=5050, debug=True)