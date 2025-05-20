from app import app, db, Notification
from sqlalchemy import inspect, Table, Column, Integer, String, Boolean, DateTime, MetaData, ForeignKey
import sqlite3
from datetime import datetime

def recreate_notification_table():
    """Drop and recreate the Notification table to ensure it has all required columns"""
    try:
        # Connect to the database
        conn = sqlite3.connect('petcare.db')
        cursor = conn.cursor()
        
        # Check if the notification table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='notification'")
        table_exists = cursor.fetchone() is not None
        
        if table_exists:
            print("Dropping existing Notification table...")
            cursor.execute("DROP TABLE notification")
            conn.commit()
            print("Notification table dropped.")
        
        # Create the notification table with all required columns
        cursor.execute("""
        CREATE TABLE notification (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            message TEXT NOT NULL,
            is_read BOOLEAN NOT NULL DEFAULT 0,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
        """)
        conn.commit()
        print("Notification table created with all required columns.")
        conn.close()
    except Exception as e:
        print(f"Error recreating notification table: {str(e)}")
        if conn:
            conn.close()

def check_and_update_columns():
    """Check if all required columns exist in the notification table"""
    inspector = inspect(db.engine)
    if inspector.has_table('notification'):
        columns = [column['name'] for column in inspector.get_columns('notification')]
        required_columns = ['id', 'user_id', 'message', 'is_read', 'created_at']
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            print(f"Missing columns in notification table: {missing_columns}")
            print("Recreating the notification table with all required columns...")
            recreate_notification_table()
        else:
            print("Notification table has all required columns.")
    else:
        print("Notification table doesn't exist. Creating it...")
        recreate_notification_table()

if __name__ == '__main__':
    with app.app_context():
        # Create all tables except notification (which we'll handle separately)
        tables_to_create = [table for table in db.metadata.tables.values() 
                           if table.name != 'notification']
        for table in tables_to_create:
            if not inspect(db.engine).has_table(table.name):
                table.create(db.engine)
                print(f"Created table: {table.name}")
        
        # Check and update the notification table
        check_and_update_columns()
        
        print("Database migration completed successfully!")
