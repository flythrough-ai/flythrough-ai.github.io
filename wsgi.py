from app import app, init_db

# Initialize the database
init_db()

# Print routes for debugging
if __name__ == "__main__":
    print("Available routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule} -> {rule.endpoint}")
    
    app.run(debug=True)