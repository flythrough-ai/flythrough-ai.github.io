from app import app, init_db
import os
from update_db import check_db_schema

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