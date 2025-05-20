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

## 🏃‍♂️ Running Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/pet-care-app.git
   cd pet-care-app
   ```

2. **Set up a virtual environment**
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with:
   ```
   SECRET_KEY=your-secret-key-here
   FLASK_APP=app.py
   FLASK_ENV=development
   DATABASE_URL=sqlite:///instance/petcare.db
   ```

5. **Initialize the database**
   ```bash
   flask init-db
   ```

6. **Run the application**
   ```bash
   flask run
   ```
   Open your browser to `http://localhost:5000`

## 📦 Dependencies

- Python 3.9+
- Flask 2.3.3
- TensorFlow 2.13.0
- OpenCV 4.8.0
- SQLAlchemy 2.0.20
- Flask-Login 0.6.2
- Gunicorn (for production)
- See `requirements.txt` for the complete list

## 🛠 Project Structure

```
pet-care-app/
├── app.py               # Main application
├── requirements.txt     # Python dependencies
├── render.yaml          # Render configuration
├── Procfile            # Process file for Render
├── wsgi.py             # WSGI entry point
├── static/             # Static files (CSS, JS, images)
│   └── uploads/        # User uploaded files
├── templates/          # HTML templates
├── instance/           # Database and instance files
└── README.md           # This file
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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
