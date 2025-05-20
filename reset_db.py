import os
import sys
import time
from pathlib import Path

# Get the absolute path to the project directory
project_dir = Path(__file__).parent.absolute()

# Set database paths
instance_dir = project_dir / 'instance'
db_path = instance_dir / 'petcare.db'
root_db_path = project_dir / 'petcare.db'

def main():
    print("=== Database Reset Tool ===\n")
    
    # Create instance directory if it doesn't exist
    if not instance_dir.exists():
        instance_dir.mkdir()
        print(f"Created instance directory: {instance_dir}")
    
    # Remove existing database files
    print("Checking for existing database files...")
    
    # Try to remove the database in the instance folder
    if db_path.exists():
        try:
            db_path.unlink()
            print(f"Removed database: {db_path}")
        except Exception as e:
            print(f"Warning: Could not remove {db_path}: {e}")
            print("The database might be in use by another application.")
            print("Please close all applications that might be using the database and try again.")
            return 1
    
    # Try to remove the database in the root folder if it exists
    if root_db_path.exists():
        try:
            root_db_path.unlink()
            print(f"Removed database: {root_db_path}")
        except Exception as e:
            print(f"Warning: Could not remove {root_db_path}: {e}")
    
    # Import the recreate_database module
    print("\nRecreating database...")
    sys.path.insert(0, str(project_dir))
    
    try:
        # Import locally to avoid circular imports
        from recreate_database import recreate_database
        recreate_database()
        
        print("\nDatabase has been successfully reset!")
        print("You can now start your Flask application.")
        print("\nTo run the application: python app.py")
        return 0
    except Exception as e:
        print(f"\nError during database recreation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
