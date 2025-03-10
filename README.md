# flythrough.ai

3D House Mesh Generator from raw video footage captured on iPhone or any camera.

## Frontend Setup

1. Start a simple server for local development:

```bash
npm start
```

This will start a basic HTTP server using Python's built-in server.

## Backend Server

The project includes a Flask-based backend server for processing videos and images.

### Backend Features

- Upload and process videos and images
- Extract frames from videos at 1-second intervals
- Store and retrieve frames with session management
- SQLite database for asset tracking
- RESTful API for frontend interaction

### Backend Setup

#### Prerequisites

- Python 3.7+
- OpenCV
- Flask

#### Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

#### Running the Backend Server

We provide a convenient script to manage the server:

```bash
# Start development server
./server.sh start dev

# Start production server (uses gunicorn)
./server.sh start

# Stop any running server
./server.sh stop

# Restart server
./server.sh restart

# Check server status
./server.sh status

# Use a different port
PORT=8080 ./server.sh start
```

The server will run at http://localhost:5000 by default.

##### Manual Starting (Alternative Methods)

```bash
# Development Mode
python app.py

# Production Mode
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

> **Note**: If you get an "Address already in use" error, use `./server.sh stop` to free the port.

#### Testing the Backend

You can test the database functionality with:

```bash
# Create test data
python test_db.py --create

# Query a specific session
python test_db.py --query <SESSION_ID>
```

### Backend API Endpoints

#### Session Management

- `POST /api/session/create` - Create a new session
- `GET /api/sessions` - List all sessions

#### Asset Management

- `POST /api/session/<session_id>/upload` - Upload a file to a session
- `GET /api/session/<session_id>/assets` - Get all assets for a session
- `GET /api/asset/<asset_id>/frames` - Get all frames for an asset
- `GET /api/file/<path:filename>` - Serve a file from the uploads directory

#### Health Check

- `GET /api/health` - Check server health

## Stripe Integration

The site integrates with Stripe for payments through:

1. Your custom backend API for creating checkout sessions
2. Stripe Checkout for payment processing
3. Stripe Pricing Table for displaying subscription options

### Stripe Setup

1. Create products and prices in the Stripe Dashboard
2. Update the pricing table ID in the `query.html` file
3. Make sure your Stripe publishable key is updated in the HTML

### Backend API Requirements

Your backend needs to implement the following API endpoint:

- `https://api.flythrough.ai/create-checkout-session` - Creates a Stripe checkout session
  - Method: POST
  - Request Body: 
    ```json
    {
      "priceId": "price_xxx",
      "productName": "Short Flythrough",
      "customerEmail": "optional_email@example.com"
    }
    ```
  - Response: 
    ```json
    {
      "id": "cs_test_xxx" // Stripe checkout session ID
    }
    ```

## Deployment

1. Deploy your static files to your preferred hosting service
2. Set up your backend API to handle the checkout session creation
3. Make sure CORS is properly configured on your API to allow requests from your domain

## Project Structure

- `index.html` - Landing page
- `query.html` - 3D visualization and purchase page
- `success.html` - Post-purchase success page
- `assets/` - Images, videos, and other static assets