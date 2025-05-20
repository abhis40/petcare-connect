# 🐾 Pet Care Application

A comprehensive pet care management system built with Flask and TensorFlow.

## 🚀 Features

- User authentication (Pet Owners & Caretakers)
- Pet profile management
- Booking system for pet care services
- Breed detection using machine learning
- Diet recommendations
- Service request management
- Real-time notifications

## 🛠️ Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Frontend**: HTML, CSS, JavaScript, Jinja2
- **Machine Learning**: TensorFlow, OpenCV
- **Database**: SQLite (development), PostgreSQL (production)
- **Deployment**: Replit

## 🚀 Deployment on Replit

### Prerequisites

- A [Replit](https://replit.com/) account (free tier is sufficient)
- Basic knowledge of Git (optional)

### Deploying to Replit

1. **Fork this repository** to your GitHub account
2. **Go to [Replit](https://replit.com/)** and sign in
3. Click the "+ Create" button and select "Import from GitHub"
4. Paste your repository URL and click "Import from GitHub"
5. Wait for Replit to import your project

### Environment Setup

1. In your Replit project, click on the "Secrets" tab (lock icon) in the left sidebar
2. Add the following environment variables:
   - `SECRET_KEY`: A random secret key for Flask sessions (generate one using `python -c 'import os; print(os.urandom(24))'`)
   - `DATABASE_URL`: `sqlite:///instance/petcare.db` (or your preferred database URL)
   - `FLASK_APP`: `app.py`
   - `FLASK_ENV`: `development` (or `production` for production)

### Running the Application

1. Once imported, Replit will automatically install dependencies from `requirements.txt`
2. The app will start automatically when you open the project
3. Click the "Run" button to start the application if it's not already running
4. Access your app using the URL provided by Replit (usually something like `https://your-project-name.your-username.repl.co`)

### Database Initialization

The database will be automatically initialized when the app starts for the first time. If you need to reset the database:

1. Open the shell in Replit
2. Run: `python init_db.py`

## 🐳 Local Development

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pet-care-app.git
   cd pet-care-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Update the values in `.env` as needed

5. Initialize the database:
   ```bash
   python init_db.py
   ```

6. Run the development server:
   ```bash
   flask run
   ```

7. Open your browser and go to `http://localhost:5000`

## 📝 Important Notes

- The free tier of Replit puts your app to sleep after a period of inactivity
- For production use, consider upgrading to a paid plan or using a different hosting service
- SQLite is used by default for development, but for production, consider using a more robust database like PostgreSQL
- Make sure to set `FLASK_ENV=production` and `DEBUG=False` in production

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [OpenCV](https://opencv.org/)
- [SQLAlchemy](https://www.sqlalchemy.org/)

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
