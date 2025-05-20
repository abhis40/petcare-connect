import sqlite3
import os
import shutil
import sys
import time
from datetime import datetime

def recreate_database():
    # Get the absolute path to the project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure instance directory exists
    instance_dir = os.path.join(project_dir, 'instance')
    if not os.path.exists(instance_dir):
        os.makedirs(instance_dir)
        print(f"Created instance directory: {instance_dir}")
    
    # Set database path in the instance folder
    db_path = os.path.join(instance_dir, 'petcare.db')
    if os.path.exists(db_path):
        # Create a backup of the database
        backup_path = os.path.join(instance_dir, f'petcare_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db')
        
        # Try to make a backup, handle if file is locked
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                shutil.copy2(db_path, backup_path)
                print(f"Created database backup at {backup_path}")
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    print(f"Database is locked. Waiting before retry {attempt+1}/{max_attempts}...")
                    time.sleep(2)  # Wait before retrying
                else:
                    print("Warning: Could not create backup as database is in use. Proceeding without backup.")
        
        # Try to delete the existing database, handle if file is locked
        try:
            os.remove(db_path)
            print(f"Removed existing database: {db_path}")
        except PermissionError:
            print("Error: Cannot remove the database as it's currently in use.")
            print("Please close any applications using the database (like Flask) and try again.")
            print("You can use Task Manager to end python.exe processes if needed.")
            return False  # Exit the function if we can't proceed
    
    # Create a new database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables with the correct schema
    print("Creating tables with correct schema...")
    
    # User table
    cursor.execute("""
    CREATE TABLE user (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email VARCHAR(100) UNIQUE NOT NULL,
        password VARCHAR(200) NOT NULL,
        name VARCHAR(100) NOT NULL,
        user_type VARCHAR(20) NOT NULL,
        location VARCHAR(100),
        joined_date DATETIME DEFAULT CURRENT_TIMESTAMP,
        phone VARCHAR(15),
        alternate_phone VARCHAR(15),
        address VARCHAR(200),
        city VARCHAR(100),
        state VARCHAR(100),
        pincode VARCHAR(10)
    )
    """)
    
    # Pet table
    cursor.execute("""
    CREATE TABLE pet (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name VARCHAR(100) NOT NULL,
        species VARCHAR(20) NOT NULL,
        breed VARCHAR(100),
        age INTEGER,
        image_filename VARCHAR(200),
        owner_id INTEGER NOT NULL,
        FOREIGN KEY (owner_id) REFERENCES user(id)
    )
    """)
    
    # CaretakerProfile table
    cursor.execute("""
    CREATE TABLE caretaker_profile (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER UNIQUE NOT NULL,
        services TEXT DEFAULT '[]',
        price_per_hour FLOAT NOT NULL,
        description VARCHAR(500),
        rating FLOAT DEFAULT 0.0,
        service_location VARCHAR(100) NOT NULL,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)
    
    # Booking table
    cursor.execute("""
    CREATE TABLE booking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pet_owner_id INTEGER NOT NULL,
        caretaker_id INTEGER NOT NULL,
        pet_id INTEGER NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME NOT NULL,
        status VARCHAR(20) DEFAULT 'pending',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (pet_owner_id) REFERENCES user(id),
        FOREIGN KEY (caretaker_id) REFERENCES user(id),
        FOREIGN KEY (pet_id) REFERENCES pet(id)
    )
    """)
    
    # Review table
    cursor.execute("""
    CREATE TABLE review (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_id INTEGER,
        service_request_id INTEGER,
        rating INTEGER NOT NULL,
        comment VARCHAR(500),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (booking_id) REFERENCES booking(id),
        FOREIGN KEY (service_request_id) REFERENCES pet_service_request(id)
    )
    """)
    
    # PetServiceRequest table
    cursor.execute("""
    CREATE TABLE pet_service_request (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pet_owner_id INTEGER NOT NULL,
        caretaker_id INTEGER,
        pet_id INTEGER NOT NULL,
        services_needed TEXT,
        start_time DATETIME NOT NULL,
        end_time DATETIME NOT NULL,
        status VARCHAR(20) DEFAULT 'open',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        description TEXT,
        FOREIGN KEY (pet_owner_id) REFERENCES user(id),
        FOREIGN KEY (caretaker_id) REFERENCES user(id),
        FOREIGN KEY (pet_id) REFERENCES pet(id)
    )
    """)
    
    # ReadNotification table
    cursor.execute("""
    CREATE TABLE read_notification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        notification_type VARCHAR(50) NOT NULL,
        notification_id INTEGER NOT NULL,
        read_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, notification_type, notification_id),
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)
    
    # CaretakerServiceOffer table
    cursor.execute("""
    CREATE TABLE caretaker_service_offer (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        caretaker_id INTEGER NOT NULL,
        service_name VARCHAR(100) NOT NULL,
        price FLOAT NOT NULL,
        description VARCHAR(500),
        FOREIGN KEY (caretaker_id) REFERENCES user(id)
    )
    """)
    
    # Notification table
    cursor.execute("""
    CREATE TABLE notification (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        message VARCHAR(255) NOT NULL,
        is_read BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user(id)
    )
    """)
    
    # Create a test user
    cursor.execute("""
    INSERT INTO user (email, password, name, user_type, location)
    VALUES ('test@example.com', 'pbkdf2:sha256:600000$5uLAGJXksZT6Jm6I$e76e5d3c2f9a75d5c0ef9e3d06f5ce4e9e82cce9c0f9a8d289a411d2b6e28a2a', 'Test User', 'pet_owner', 'Test Location')
    """)
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    print("Database recreation completed successfully.")
    return True

def main():
    print("Starting database recreation process...")
    try:
        success = recreate_database()
        if success is False:
            print("\nDatabase recreation failed. Please ensure no other applications are using the database.")
            print("Tip: You may need to stop any running Flask applications first.")
            sys.exit(1)
        print("\nDatabase has been successfully recreated and is ready to use.")
        return 0
    except Exception as e:
        print(f"\nError during database recreation: {str(e)}")
        print("Please fix the issue and try again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
