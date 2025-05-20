#!/usr/bin/env python

import os
import sys
import subprocess
import time

# Get the absolute path to the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))

def kill_python_processes():
    """Attempt to kill any running Python processes"""
    print("Checking for running Python processes...")
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'python.exe'], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Attempted to stop Python processes")
            time.sleep(1)  # Give processes time to terminate
        else:  # Unix-like
            subprocess.run(['pkill', '-f', 'python'], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("Attempted to stop Python processes")
            time.sleep(1)  # Give processes time to terminate
    except Exception as e:
        print(f"Warning: Failed to kill processes: {e}")

def main():
    print("=== Database Initialization Tool ===")
    print("This tool will reset your database to its initial state.")
    print("WARNING: All existing data will be lost!")
    
    # Kill any running Python processes
    kill_python_processes()
    
    # Import and run the database recreation script
    print("\nRecreating database...")
    sys.path.insert(0, project_dir)
    
    try:
        from recreate_database import recreate_database
        result = recreate_database()
        
        if result is False:
            print("\nFailed to recreate database. Please ensure no applications are using it.")
            return 1
            
        print("\nDatabase has been successfully initialized!")
        print("You can now start your Flask application.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
