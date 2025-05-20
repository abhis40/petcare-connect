# Pet Care Application

## Database Setup and Management

This project uses SQLite for data storage. The database file is located in the `instance` folder.

### Setting Up the Database

To initialize or reset the database:

1. Make sure no Flask application is running
2. Run the database initialization script:

```
python init_db.py
```

This script will:
- Stop any running Flask processes (if possible)
- Create a backup of the existing database (if one exists)
- Recreate the database with the correct schema

### Running the Application

To run the application:

```
python app.py
```

The application will be available at http://127.0.0.1:5000

### Troubleshooting Database Issues

If you encounter database-related errors:

1. Stop any running Flask applications
2. Run `python init_db.py` to reset the database
3. If you still have issues, manually end Python processes through Task Manager (Windows) or Activity Monitor (Mac)

### Database Structure

The application uses the following main tables:
- `user`: Stores user information
- `pet`: Stores pet information
- `caretaker_profile`: Stores caretaker-specific information
- `booking`: Stores booking information
- `pet_service_request`: Stores service requests
- `review`: Stores reviews

## VS Code Tips

- Use the integrated terminal in VS Code to run the scripts
- If you get database lock errors, use the Terminal menu to kill tasks
- Install the SQLite extension for VS Code to easily view and edit the database
