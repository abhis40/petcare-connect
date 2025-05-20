"""
WSGI config for Pet Care Application.

This module contains the WSGI callable for the application.
"""

from app import app, db

if __name__ == "__main__":
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=False)
