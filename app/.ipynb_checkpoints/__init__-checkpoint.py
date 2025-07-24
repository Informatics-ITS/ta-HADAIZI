from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
import os

# Create output directories if they don't exist
os.makedirs('temp_jobs', exist_ok=True)
os.makedirs('output_images', exist_ok=True)

# Initialize SocketIO with CORS allowed
socketio = SocketIO(cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    
    # Enable CORS for the Flask app
    CORS(app)
    
    # Register blueprint
    from app.routes import ergonomic_bp
    app.register_blueprint(ergonomic_bp)
    
    # Initialize SocketIO with the app
    socketio.init_app(app)
    
    return app
