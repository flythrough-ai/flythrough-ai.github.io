#!/bin/bash

# Server management script for flythrough.ai
PORT="${PORT:-5000}"

# Function to check if a process is running on the port
check_port() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -i :"$PORT" >/dev/null 2>&1
    return $?
  else
    # Fallback for systems without lsof
    netstat -tuln | grep -q ":$PORT "
    return $?
  fi
}

# Function to kill processes on a port
kill_port() {
  echo "Stopping processes on port $PORT..."
  if command -v lsof >/dev/null 2>&1; then
    # Get PIDs using lsof and kill them
    lsof -i :"$PORT" -t | xargs -r kill -9
  else
    # Fallback using more complex grep and awk
    PIDs=$(netstat -tuln | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1)
    if [ -n "$PIDs" ]; then
      echo "$PIDs" | xargs -r kill -9
    fi
  fi
  sleep 1
  if ! check_port; then
    echo "Port $PORT is now free"
  else
    echo "WARNING: Could not free port $PORT"
    return 1
  fi
}

case "$1" in
  start)
    if check_port; then
      echo "Port $PORT is already in use. Use 'stop' first or specify a different port with PORT=xxxx"
      exit 1
    fi
    
    # Check if we're using development or production mode
    if [ "$2" = "dev" ]; then
      echo "Starting development server on port $PORT..."
      # Set Flask to debug mode for better error messages
      export FLASK_ENV=development
      export FLASK_DEBUG=1
      python3 app.py
    else
      echo "Starting production server on port $PORT..."
      # Increase Gunicorn verbosity for debugging
      gunicorn --bind 0.0.0.0:"$PORT" wsgi:app --log-level debug
    fi
    ;;
    
  stop)
    kill_port
    ;;
    
  restart)
    kill_port && sleep 1
    if [ "$2" = "dev" ]; then
      echo "Restarting development server on port $PORT..."
      python app.py
    else
      echo "Restarting production server on port $PORT..."
      gunicorn --bind 0.0.0.0:"$PORT" wsgi:app
    fi
    ;;
    
  status)
    if check_port; then
      echo "Server is running on port $PORT"
      if command -v lsof >/dev/null 2>&1; then
        echo "Process details:"
        lsof -i :"$PORT"
      fi
    else
      echo "No server running on port $PORT"
    fi
    ;;
    
  *)
    echo "Usage: $0 {start|stop|restart|status} [dev]"
    echo ""
    echo "Examples:"
    echo "  $0 start       # Start production server"
    echo "  $0 start dev   # Start development server"
    echo "  PORT=8000 $0 start  # Start server on port 8000"
    exit 1
    ;;
esac
