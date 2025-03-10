from app import app, init_db

# Initialize the database
init_db()

if __name__ == "__main__":
    app.run()