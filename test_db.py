import sqlite3
import os
import sys
import uuid
import time

# Path to the database
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def init_db():
    """Initialize the database with test data."""
    # Initialize database structure
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
        FOREIGN KEY (asset_id) REFERENCES assets (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database structure initialized.")
    
def create_test_data():
    """Insert some test data."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create test session
    session_id = str(uuid.uuid4())
    cursor.execute(
        "INSERT INTO sessions (id, created_at, name, description) VALUES (?, ?, ?, ?)",
        (session_id, int(time.time()), "Test Session", "A test session for database validation")
    )
    
    # Create test assets
    asset_types = [
        {"name": "test_video.mp4", "type": "mp4"},
        {"name": "test_image.jpg", "type": "jpg"}
    ]
    
    asset_ids = []
    for asset in asset_types:
        asset_id = str(uuid.uuid4())
        asset_ids.append(asset_id)
        
        cursor.execute(
            "INSERT INTO assets (id, session_id, original_filename, file_type, upload_time, processed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (asset_id, session_id, asset["name"], asset["type"], int(time.time()), True)
        )
    
    # Create test frames
    for i, asset_id in enumerate(asset_ids):
        # For the video asset, create multiple frames
        if i == 0:
            for j in range(10):
                frame_id = str(uuid.uuid4())
                frame_path = f"/test/path/to/frame_{j:04d}.jpg"
                
                cursor.execute(
                    "INSERT INTO frames (id, asset_id, frame_number, file_path, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (frame_id, asset_id, j, frame_path, j * 1.0)
                )
        # For the image asset, create one frame
        else:
            frame_id = str(uuid.uuid4())
            frame_path = f"/test/path/to/{asset_types[i]['name']}"
            
            cursor.execute(
                "INSERT INTO frames (id, asset_id, frame_number, file_path, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (frame_id, asset_id, 0, frame_path, 0.0)
            )
    
    conn.commit()
    conn.close()
    
    print(f"Test data created. Session ID: {session_id}")
    return session_id

def query_test_data(session_id):
    """Query and display test data."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # This enables column access by name
    cursor = conn.cursor()
    
    # Query session info
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session = cursor.fetchone()
    
    if not session:
        print(f"Session {session_id} not found!")
        conn.close()
        return
    
    print("\n=== Session Info ===")
    print(f"ID: {session['id']}")
    print(f"Name: {session['name']}")
    print(f"Created at: {session['created_at']}")
    print(f"Description: {session['description']}")
    
    # Query assets for this session
    cursor.execute("SELECT * FROM assets WHERE session_id = ?", (session_id,))
    assets = cursor.fetchall()
    
    print(f"\n=== Assets ({len(assets)}) ===")
    for asset in assets:
        print(f"\nAsset ID: {asset['id']}")
        print(f"Filename: {asset['original_filename']}")
        print(f"Type: {asset['file_type']}")
        print(f"Processed: {asset['processed']}")
        
        # Query frames for this asset
        cursor.execute("SELECT * FROM frames WHERE asset_id = ? ORDER BY frame_number", (asset['id'],))
        frames = cursor.fetchall()
        
        print(f"Frames: {len(frames)}")
        for frame in frames:
            print(f"  Frame {frame['frame_number']}: {frame['file_path']} (t={frame['timestamp']}s)")
    
    conn.close()

def main():
    """Main function."""
    if not os.path.exists(DATABASE_PATH):
        print("Database doesn't exist. Creating...")
        init_db()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create":
        session_id = create_test_data()
        query_test_data(session_id)
    elif len(sys.argv) > 2 and sys.argv[1] == "--query":
        session_id = sys.argv[2]
        query_test_data(session_id)
    else:
        print("Usage:")
        print("  python test_db.py --create           # Create test data")
        print("  python test_db.py --query SESSION_ID # Query test data for a session")

if __name__ == "__main__":
    main()