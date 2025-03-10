import sqlite3
import os

# Path to the database
DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database.db')

def check_db_schema():
    """Check the database schema and update it if needed."""
    print(f"Checking database schema at {DATABASE_PATH}...")
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if frames table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='frames'")
    if cursor.fetchone() is None:
        print("Frames table doesn't exist. Creating...")
        cursor.execute('''
        CREATE TABLE frames (
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
        print("Frames table created.")
    else:
        # Check if camera_pose column exists
        try:
            cursor.execute("SELECT camera_pose FROM frames LIMIT 1")
            print("camera_pose column exists.")
        except sqlite3.OperationalError:
            print("Adding camera_pose column to frames table...")
            cursor.execute("ALTER TABLE frames ADD COLUMN camera_pose TEXT")
            conn.commit()
        
        # Check if point_cloud_data column exists
        try:
            cursor.execute("SELECT point_cloud_data FROM frames LIMIT 1")
            print("point_cloud_data column exists.")
        except sqlite3.OperationalError:
            print("Adding point_cloud_data column to frames table...")
            cursor.execute("ALTER TABLE frames ADD COLUMN point_cloud_data TEXT")
            conn.commit()
    
    # Check if sessions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'")
    if cursor.fetchone() is None:
        print("Sessions table doesn't exist. Creating...")
        cursor.execute('''
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            created_at INTEGER,
            name TEXT,
            description TEXT
        )
        ''')
        conn.commit()
        print("Sessions table created.")
    
    # Check if assets table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='assets'")
    if cursor.fetchone() is None:
        print("Assets table doesn't exist. Creating...")
        cursor.execute('''
        CREATE TABLE assets (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            original_filename TEXT,
            file_type TEXT,
            upload_time INTEGER,
            processed BOOLEAN,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
        ''')
        conn.commit()
        print("Assets table created.")
    
    conn.close()
    print("Database schema check complete.")

def main():
    """Main function."""
    check_db_schema()
    print("Database update complete.")

if __name__ == "__main__":
    main()