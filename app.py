from flask import Flask, request, jsonify, send_from_directory, make_response
from flask_cors import CORS
import os
import uuid
import time
import cv2
import sqlite3
from werkzeug.utils import secure_filename
import shutil
import logging
import atexit
import threading
from PIL import Image
import io
import base64
import json
import numpy as np
import math
import random

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)  # Enable CORS for all routes

# Configuration
# Check if we're on Render to use the persistent disk location
if os.environ.get('RENDER'):
    UPLOAD_FOLDER = '/opt/render/project/data/uploads'
    THUMBNAIL_FOLDER = '/opt/render/project/data/thumbs'
    DATABASE_PATH = '/opt/render/project/data/database.db'
else:
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    THUMBNAIL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbs')
    DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 1024 * 1024 * 1024  # 1GB max upload size
SESSION_TIMEOUT = 3600  # Session timeout in seconds (1 hour)
THUMBNAIL_WIDTH = 142  # Match the width in editor.html
THUMBNAIL_HEIGHT = 80  # Match the height in editor.html

# Create required folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['THUMBNAIL_FOLDER'] = THUMBNAIL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# In-memory thumbnail cache
thumbnail_cache = {}

# Active sessions tracker
active_sessions = set()
session_last_activity = {}
session_lock = threading.Lock()

# Database initialization
def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at INTEGER,
        name TEXT,
        description TEXT
    )
    ''')
    
    # Create assets table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        id TEXT PRIMARY KEY,
        session_id TEXT,
        original_filename TEXT,
        file_type TEXT,
        upload_time INTEGER,
        processed BOOLEAN,
        FOREIGN KEY (session_id) REFERENCES sessions (id)
    )
    ''')
    
    # Create frames table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS frames (
        id TEXT PRIMARY KEY,
        asset_id TEXT,
        frame_number INTEGER,
        file_path TEXT,
        timestamp REAL,
        camera_pose TEXT,
        point_cloud_data TEXT,
        FOREIGN KEY (asset_id) REFERENCES assets (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized")

# Session management functions
def register_session_activity(session_id):
    """Register activity for a session to prevent cleanup."""
    with session_lock:
        active_sessions.add(session_id)
        session_last_activity[session_id] = time.time()

def cleanup_inactive_sessions():
    """Clean up inactive sessions both from memory and database."""
    current_time = time.time()
    sessions_to_remove = []
    
    with session_lock:
        for session_id, last_activity in session_last_activity.items():
            if current_time - last_activity > SESSION_TIMEOUT:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            active_sessions.discard(session_id)
            session_last_activity.pop(session_id, None)
            
            # Also clear any cached thumbnails for this session
            keys_to_remove = [k for k in thumbnail_cache.keys() if k.startswith(f"{session_id}/")]
            for key in keys_to_remove:
                thumbnail_cache.pop(key, None)
    
    # Now clean up the database and files for inactive sessions
    if sessions_to_remove:
        cleanup_session_data(sessions_to_remove)

def cleanup_session_data(session_ids):
    """Clean up session data from database and files."""
    if not session_ids:
        return
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for session_id in session_ids:
        try:
            # Get all assets for this session
            cursor.execute("SELECT id FROM assets WHERE session_id = ?", (session_id,))
            asset_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete frames for these assets
            if asset_ids:
                placeholders = ', '.join(['?'] * len(asset_ids))
                cursor.execute(f"DELETE FROM frames WHERE asset_id IN ({placeholders})", asset_ids)
            
            # Delete assets
            cursor.execute("DELETE FROM assets WHERE session_id = ?", (session_id,))
            
            # Delete session
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            # Remove session directory
            session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                
            logger.info(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.error(f"Error cleaning up session {session_id}: {str(e)}")
    
    conn.commit()
    conn.close()

# Schedule periodic cleanup
def schedule_cleanup():
    cleanup_inactive_sessions()
    # Schedule next cleanup in 15 minutes
    threading.Timer(900, schedule_cleanup).start()

# Start cleanup scheduler
threading.Timer(900, schedule_cleanup).start()

# Register cleanup on application exit
@atexit.register
def cleanup_on_exit():
    """Clean up all sessions on application exit."""
    with session_lock:
        cleanup_session_data(list(active_sessions))

# Thumbnail generation and caching
def generate_thumbnail(image_path, thumb_size=(THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT)):
    """Generate a thumbnail from an image file."""
    try:
        # Check if thumbnail is already in cache
        if image_path in thumbnail_cache:
            return thumbnail_cache[image_path]
        
        # Generate thumbnail
        img = Image.open(image_path)
        img.thumbnail(thumb_size, Image.LANCZOS)
        
        # Save to memory buffer
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        
        # Cache the thumbnail
        thumbnail_data = buffer.getvalue()
        thumbnail_cache[image_path] = thumbnail_data
        
        return thumbnail_data
    except Exception as e:
        logger.error(f"Error generating thumbnail for {image_path}: {str(e)}")
        return None

def get_thumbnail_path(frame_path):
    """Get thumbnail path for a frame."""
    thumb_dir = app.config['THUMBNAIL_FOLDER']
    
    # Get the relative path from the upload folder
    try:
        frame_rel_path = os.path.relpath(frame_path, app.config['UPLOAD_FOLDER'])
    except ValueError:
        # Handle case when frame_path is not under app.config['UPLOAD_FOLDER']
        # Use a reasonable fallback
        frame_rel_path = os.path.basename(frame_path)
        logger.warning(f"Frame path {frame_path} is not under upload folder")
    
    thumb_path = os.path.join(thumb_dir, frame_rel_path)
    
    # Ensure the thumbnail directory exists
    os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
    
    logger.info(f"Thumbnail path: {thumb_path}")
    
    return thumb_path

# Functions for generating 3D data
def generate_random_point_cloud(num_points=50, frame_number=0, total_frames=20):
    """Generate random point cloud data."""
    # Define a progress parameter (0 to 1) for smooth camera path generation
    progress = frame_number / max(1, total_frames - 1)
    
    # Create some structure in the point cloud
    # Points will form a curved shape that evolves over frames
    points = []
    for i in range(num_points):
        # Basic point position 
        angle = (i / num_points) * math.pi * 2
        radius = 5 + random.uniform(-1, 1)
        
        # Add some variation based on frame number
        x_offset = math.sin(progress * math.pi) * 3
        y_offset = math.cos(progress * math.pi * 0.5) * 2
        z_offset = (progress - 0.5) * 4
        
        x = radius * math.cos(angle) + x_offset + random.uniform(-0.2, 0.2)
        y = radius * math.sin(angle) + y_offset + random.uniform(-0.2, 0.2)
        z = random.uniform(-1, 1) + z_offset
        
        # Add a color (RGB values between 0 and 1)
        color_progress = (angle / (math.pi * 2) + progress) % 1.0
        r = 0.5 + 0.5 * math.sin(color_progress * math.pi * 2)
        g = 0.5 + 0.5 * math.sin((color_progress + 0.33) * math.pi * 2)
        b = 0.5 + 0.5 * math.sin((color_progress + 0.67) * math.pi * 2)
        
        # Create a point with position and color
        point = {
            "position": [float(x), float(y), float(z)],
            "color": [float(r), float(g), float(b)]
        }
        
        points.append(point)
    
    # Add one "special" large point that stands out
    special_angle = progress * math.pi * 2
    special_x = 4 * math.cos(special_angle)
    special_y = 4 * math.sin(special_angle)
    special_z = math.sin(progress * math.pi * 4) * 2
    
    special_point = {
        "position": [float(special_x), float(special_y), float(special_z)],
        "color": [1.0, 0.5, 0.0],
        "size": 2.0,  # Larger point
        "isEmphasis": True
    }
    
    points.append(special_point)
    
    return points

def generate_camera_pose(frame_number=0, total_frames=20):
    """Generate camera pose data including position, rotation, and view frustum."""
    # Define a progress parameter (0 to 1) for smooth camera path
    progress = frame_number / max(1, total_frames - 1)
    
    # Generate a circular path for the camera
    angle = progress * math.pi * 2
    radius = 8
    
    # Camera position
    pos_x = radius * math.cos(angle)
    pos_y = radius * math.sin(angle)
    pos_z = 2 * math.sin(progress * math.pi * 2) + 3
    
    # Camera is looking at the origin with some offset
    look_x = 0
    look_y = 0
    look_z = math.sin(progress * math.pi) * 2
    
    # Calculate the direction vector
    direction_x = look_x - pos_x
    direction_y = look_y - pos_y
    direction_z = look_z - pos_z
    
    # Normalize the direction vector
    length = math.sqrt(direction_x**2 + direction_y**2 + direction_z**2)
    direction_x /= length
    direction_y /= length
    direction_z /= length
    
    # Define up vector (usually +Z in our case)
    up_x = 0
    up_y = 0
    up_z = 1
    
    # Frustum parameters - much smaller frustum
    fov = 45  # Field of view in degrees
    aspect_ratio = 16/9
    near = 0.01  # 10x smaller near plane
    far = 10     # 10x smaller far plane
    
    # Package the camera data
    camera_data = {
        "position": [float(pos_x), float(pos_y), float(pos_z)],
        "direction": [float(direction_x), float(direction_y), float(direction_z)],
        "up": [float(up_x), float(up_y), float(up_z)],
        "frustum": {
            "fov": float(fov),
            "aspect": float(aspect_ratio),
            "near": float(near),
            "far": float(far)
        }
    }
    
    return camera_data

# Helper function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to extract frames from video
def extract_frames(video_path, output_folder, frame_interval=1.0):
    """
    Extract frames from a video file at specified interval.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder to save extracted frames
        frame_interval: Interval in seconds between frames
    
    Returns:
        List of dictionaries with frame info (number, path, timestamp)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    logger.info(f"Video properties: FPS={fps}, Total Frames={total_frames}, Duration={duration}s")
    
    frames_info = []
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate current timestamp
        timestamp = frame_count / fps if fps > 0 else 0
        
        # Save frame at the specified interval
        if timestamp >= saved_count * frame_interval:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            
            # Save full-size frame
            cv2.imwrite(frame_path, frame)
            
            # Generate thumbnail
            thumb_path = get_thumbnail_path(frame_path)
            # Convert from BGR to RGB for PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            pil_img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.LANCZOS)
            pil_img.save(thumb_path, "JPEG", quality=85)
            
            # Cache the thumbnail in memory
            with open(thumb_path, "rb") as f:
                thumbnail_cache[frame_path] = f.read()
            
            # Generate 3D data for this frame
            point_cloud = generate_random_point_cloud(
                num_points=100,  # Increased number of points
                frame_number=saved_count,
                total_frames=int(duration / frame_interval)
            )
            
            camera_pose = generate_camera_pose(
                frame_number=saved_count,
                total_frames=int(duration / frame_interval)
            )
            
            frames_info.append({
                "frame_number": saved_count,
                "file_path": frame_path,
                "timestamp": timestamp,
                "thumb_path": thumb_path,
                "point_cloud_data": json.dumps(point_cloud),
                "camera_pose": json.dumps(camera_pose)
            })
            
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    logger.info(f"Extracted {saved_count} frames from video")
    return frames_info

# Helper function to process an image file
def process_image(image_path, output_folder):
    """
    Process a single image file.
    
    Args:
        image_path: Path to the image file
        output_folder: Folder to save processed image
    
    Returns:
        Dictionary with frame info
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Error reading image file: {image_path}")
        return None
    
    # Save a copy of the image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img)
    
    # Generate thumbnail
    thumb_path = get_thumbnail_path(output_path)
    # Convert from BGR to RGB for PIL
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    pil_img.thumbnail((THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT), Image.LANCZOS)
    pil_img.save(thumb_path, "JPEG", quality=85)
    
    # Cache the thumbnail in memory
    with open(thumb_path, "rb") as f:
        thumbnail_cache[output_path] = f.read()
    
    # Generate 3D data for single image
    point_cloud = generate_random_point_cloud(
        num_points=100,
        frame_number=0,
        total_frames=1
    )
    
    camera_pose = generate_camera_pose(
        frame_number=0,
        total_frames=1
    )
    
    return {
        "frame_number": 0,
        "file_path": output_path,
        "timestamp": 0.0,
        "thumb_path": thumb_path,
        "point_cloud_data": json.dumps(point_cloud),
        "camera_pose": json.dumps(camera_pose)
    }

# Routes
@app.route('/')
def serve_index():
    """Serve the index.html file."""
    return app.send_static_file('index.html')

@app.route('/editor.html')
def serve_editor():
    """Serve the editor.html file."""
    return app.send_static_file('editor.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    # Log current active sessions for debugging
    with session_lock:
        active_session_count = len(active_sessions)
        logger.info(f"Health check: {active_session_count} active sessions")
        if active_session_count > 0:
            logger.info(f"Active sessions: {', '.join(list(active_sessions))}")
    
    return jsonify({
        "status": "healthy", 
        "timestamp": time.time(),
        "active_sessions": list(active_sessions)
    })

@app.route('/api/session/create', methods=['GET', 'POST'])
def create_session():
    """Create a new session."""
    logger.info(f"Session create request received: method={request.method}, path={request.path}")
    
    session_id = str(uuid.uuid4())
    
    # Handle both GET and POST methods
    if request.method == 'POST':
        name = request.form.get('name', f"Session {session_id[:8]}")
        description = request.form.get('description', '')
        logger.info(f"POST request form data: {dict(request.form)}")
    else:  # GET method
        name = request.args.get('name', f"Session {session_id[:8]}")
        description = request.args.get('description', '')
        logger.info(f"GET request args: {dict(request.args)}")
    
    try:
        # Register this as an active session
        register_session_activity(session_id)
        
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO sessions (id, created_at, name, description) VALUES (?, ?, ?, ?)",
            (session_id, int(time.time()), name, description)
        )
        
        conn.commit()
        conn.close()
        
        # Create session directory
        session_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Create thumbnail directory for this session
        thumb_dir = os.path.join(app.config['THUMBNAIL_FOLDER'], session_id)
        os.makedirs(thumb_dir, exist_ok=True)
        
        # Log success
        logger.info(f"Created new session successfully: {session_id}")
        
        return jsonify({
            "session_id": session_id,
            "name": name,
            "created_at": int(time.time())
        })
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return jsonify({"error": f"Failed to create session: {str(e)}"}), 500

@app.route('/api/session/<session_id>/upload', methods=['POST'])
def upload_file(session_id):
    """Upload a file to a session."""
    # Register activity for this session
    register_session_activity(session_id)
    
    # Check if session exists
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT id FROM sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    
    if not session:
        conn.close()
        return jsonify({"error": "Session not found"}), 404
    
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    # Generate asset ID and save file
    asset_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    file_type = filename.rsplit('.', 1)[1].lower()
    
    # Create asset directory
    asset_dir = os.path.join(app.config['UPLOAD_FOLDER'], session_id, asset_id)
    os.makedirs(asset_dir, exist_ok=True)
    
    # Save original file
    file_path = os.path.join(asset_dir, filename)
    file.save(file_path)
    
    # Record asset in database
    cursor.execute(
        "INSERT INTO assets (id, session_id, original_filename, file_type, upload_time, processed) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (asset_id, session_id, filename, file_type, int(time.time()), False)
    )
    conn.commit()
    
    # Process file based on type
    frames_dir = os.path.join(asset_dir, "frames")
    
    try:
        frames_info = []
        
        # Process video files
        if file_type in ['mp4', 'avi', 'mov']:
            frames_info = extract_frames(file_path, frames_dir)
        
        # Process image files
        elif file_type in ['jpg', 'jpeg', 'png']:
            frame_info = process_image(file_path, frames_dir)
            if frame_info:
                frames_info = [frame_info]
        
        # Record frames in database
        for frame in frames_info:
            frame_id = str(uuid.uuid4())
            cursor.execute(
                "INSERT INTO frames (id, asset_id, frame_number, file_path, timestamp, camera_pose, point_cloud_data) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    frame_id, 
                    asset_id, 
                    frame["frame_number"], 
                    frame["file_path"], 
                    frame["timestamp"],
                    frame.get("camera_pose", "{}"),
                    frame.get("point_cloud_data", "[]")
                )
            )
        
        # Mark asset as processed
        cursor.execute(
            "UPDATE assets SET processed = ? WHERE id = ?",
            (True, asset_id)
        )
        
        conn.commit()
        
        response = {
            "asset_id": asset_id,
            "session_id": session_id,
            "filename": filename,
            "frames_count": len(frames_info)
        }
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error processing file: {str(e)}")
        response = {"error": f"Error processing file: {str(e)}"}
    
    conn.close()
    return jsonify(response)

@app.route('/api/session/<session_id>/assets', methods=['GET'])
def get_session_assets(session_id):
    """Get all assets for a session."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, original_filename, file_type, upload_time, processed "
        "FROM assets WHERE session_id = ?",
        (session_id,)
    )
    
    assets = [dict(row) for row in cursor.fetchall()]
    
    for asset in assets:
        # Get frame count for each asset
        cursor.execute(
            "SELECT COUNT(*) as frame_count FROM frames WHERE asset_id = ?",
            (asset['id'],)
        )
        frame_count = cursor.fetchone()
        asset['frame_count'] = frame_count['frame_count'] if frame_count else 0
    
    conn.close()
    return jsonify({"session_id": session_id, "assets": assets})

@app.route('/api/asset/<asset_id>/frames', methods=['GET'])
def get_asset_frames(asset_id):
    """Get all frames for an asset."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get asset info
    cursor.execute(
        "SELECT id, session_id, original_filename FROM assets WHERE id = ?",
        (asset_id,)
    )
    asset = cursor.fetchone()
    
    if not asset:
        conn.close()
        return jsonify({"error": "Asset not found"}), 404
        
    # Register activity for this session
    register_session_activity(asset['session_id'])
    
    # Get frames with 3D data
    cursor.execute(
        "SELECT id, frame_number, file_path, timestamp, camera_pose, point_cloud_data "
        "FROM frames WHERE asset_id = ? ORDER BY frame_number",
        (asset_id,)
    )
    
    frames = []
    for row in cursor.fetchall():
        frame_data = dict(row)
        # Convert file path to URL
        file_path = frame_data['file_path']
        rel_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
        frame_data['url'] = f"/api/file/{rel_path}"
        frame_data['thumbnail_url'] = f"/api/thumbnail/{rel_path}"
        frames.append(frame_data)
    
    conn.close()
    return jsonify({
        "asset_id": asset_id,
        "session_id": asset['session_id'],
        "filename": asset['original_filename'],
        "frames": frames
    })

@app.route('/api/file/<path:filename>', methods=['GET'])
def serve_file(filename):
    """Serve a file from the uploads directory."""
    # Register activity for the session to prevent cleanup
    # Extract session_id from path (first component of path)
    parts = filename.split('/')
    if len(parts) > 0:
        register_session_activity(parts[0])
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/thumbnail/<path:filename>', methods=['GET'])
def serve_thumbnail(filename):
    """Serve a thumbnail for a file."""
    # Extract session_id from path (first component of path)
    parts = filename.split('/')
    if len(parts) > 0:
        register_session_activity(parts[0])
    
    # Construct full file path for the original image
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if we have a cached thumbnail
    if file_path in thumbnail_cache:
        response = make_response(thumbnail_cache[file_path])
        response.headers.set('Content-Type', 'image/jpeg')
        return response
    
    # Generate thumbnail on-demand if not in cache
    thumb_path = get_thumbnail_path(file_path)
    
    # If thumbnail file exists, serve it
    if os.path.exists(thumb_path):
        # Cache it for future use
        with open(thumb_path, "rb") as f:
            thumbnail_cache[file_path] = f.read()
        
        return send_from_directory(os.path.dirname(thumb_path), os.path.basename(thumb_path))
    
    # If thumbnail doesn't exist, generate it
    thumbnail_data = generate_thumbnail(file_path)
    
    if thumbnail_data:
        # Save to file
        os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
        with open(thumb_path, "wb") as f:
            f.write(thumbnail_data)
        
        response = make_response(thumbnail_data)
        response.headers.set('Content-Type', 'image/jpeg')
        return response
    
    # If all fails, try to send the original
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/session/<session_id>/close', methods=['POST'])
def close_session(session_id):
    """Close a session and clean up its resources."""
    # Remove session from active sessions
    with session_lock:
        active_sessions.discard(session_id)
        session_last_activity.pop(session_id, None)
    
    # Clean up session data
    cleanup_session_data([session_id])
    
    return jsonify({"status": "success", "message": f"Session {session_id} has been closed"})

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        "SELECT id, created_at, name, description FROM sessions ORDER BY created_at DESC"
    )
    
    sessions = [dict(row) for row in cursor.fetchall()]
    
    for session in sessions:
        # Get asset count for each session
        cursor.execute(
            "SELECT COUNT(*) as asset_count FROM assets WHERE session_id = ?",
            (session['id'],)
        )
        asset_count = cursor.fetchone()
        session['asset_count'] = asset_count['asset_count'] if asset_count else 0
    
    conn.close()
    return jsonify({"sessions": sessions})

if __name__ == '__main__':
    # Initialize the database
    init_db()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)