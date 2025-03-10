from app import app, init_db
import os
import sqlite3
from update_db import check_db_schema

# Ensure the required directories exist
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
thumbs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'thumbs')
os.makedirs(uploads_dir, exist_ok=True)
os.makedirs(thumbs_dir, exist_ok=True)

# Check database schema and update if needed
try:
    check_db_schema()
except Exception as e:
    print(f"Error checking database schema: {e}")

# Initialize the database
try:
    init_db()
except Exception as e:
    print(f"Error initializing database: {e}")

# Print routes for debugging
if __name__ == "__main__":
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule} -> {rule.endpoint}")
    
    app.run(debug=True)