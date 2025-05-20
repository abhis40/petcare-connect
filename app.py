from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import cv2
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime, timedelta
import calendar

import os
import sys
from pathlib import Path

# Ensure required directories exist
REQUIRED_DIRS = [
    'static/uploads',
    'instance'
]

for directory in REQUIRED_DIRS:
    Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize Flask application
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configuration
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        import secrets
        SECRET_KEY = secrets.token_hex(24)
        print(f"WARNING: Using auto-generated SECRET_KEY. Set SECRET_KEY in production!")
    
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///instance/petcare.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_recycle': 300,
        'pool_pre_ping': True,
    }

# Apply configuration
app.config.from_object(Config)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Notification model
class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    message = db.Column(db.String(255), nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref=db.backref('notifications', lazy=True))
    
    def __repr__(self):
        return f'<Notification {self.id}>'

# Helper function to get notifications for the current user
def get_notifications():
    """Get notifications for the current user."""
    if not current_user.is_authenticated:
        return {'unread_notifications': 0, 'all_notifications': []}
    
    # Get all notifications for the current user
    all_notifications = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).all()
    
    # Count unread notifications
    unread_notifications = sum(1 for notification in all_notifications if not notification.is_read)
    
    return {
        'unread_notifications': unread_notifications,
        'all_notifications': all_notifications
    }

def create_notification(user_id, message):
    """Create a new notification for a user."""
    try:
        notification = Notification(
            user_id=user_id,
            message=message,
            is_read=False
        )
        db.session.add(notification)
        db.session.commit()
        return True
    except Exception as e:
        app.logger.error(f"Error creating notification: {str(e)}")
        return False

# Initialize login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load models
dog_model = None
cat_model = None


# Custom DepthwiseConv2D class to handle version incompatibility
class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1,
                 data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True,
                 depthwise_initializer='glorot_uniform', bias_initializer='zeros',
                 depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 depthwise_constraint=None, bias_constraint=None, **kwargs):
        # Remove 'groups' if it exists in kwargs
        if 'groups' in kwargs:
            del kwargs['groups']

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )


# Function to load the TensorFlow models
def load_models():
    global dog_model, cat_model
    try:
        # Create custom objects to handle the compatibility issue
        custom_objects = {
            'DepthwiseConv2D': CustomDepthwiseConv2D
        }

        try:
            # Update these paths to where your .h5 model files are stored
            dog_model_path = 'models/dog_model.h5'
            if os.path.exists(dog_model_path):
                dog_model = tf.keras.models.load_model(dog_model_path, custom_objects=custom_objects, compile=False)
                print("Dog model loaded successfully!")
            else:
                print(f"Dog model file not found at {dog_model_path}")
                dog_model = None
        except Exception as e:
            print(f"Error loading dog model: {e}")
            dog_model = None

        try:
            cat_model_path = 'models/cat_model.h5'
            if os.path.exists(cat_model_path):
                cat_model = tf.keras.models.load_model(cat_model_path, custom_objects=custom_objects, compile=False)
                print("Cat model loaded successfully!")
            else:
                print(f"Cat model file not found at {cat_model_path}")
                cat_model = None
        except Exception as e:
            print(f"Error loading cat model: {e}")
            cat_model = None

        # Check if at least one model loaded
        if dog_model is None and cat_model is None:
            print("Warning: No models were loaded. Breed identification will not work.")

    except Exception as e:
        print(f"Error in load_models: {e}")


# Class definitions for database models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)  # 'pet_owner' or 'caretaker'
    location = db.Column(db.String(100))
    joined_date = db.Column(db.DateTime, default=datetime.utcnow)
    phone = db.Column(db.String(15))
    alternate_phone = db.Column(db.String(15))
    address = db.Column(db.String(200))
    city = db.Column(db.String(100))
    state = db.Column(db.String(100))
    pincode = db.Column(db.String(10))

    # Relationships
    pets = db.relationship('Pet', backref='owner', lazy=True)
    caretaker_profile = db.relationship('CaretakerProfile', backref='user', lazy=True, uselist=False)


class Pet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    species = db.Column(db.String(20), nullable=False)  # 'dog' or 'cat'
    breed = db.Column(db.String(100))
    age = db.Column(db.Integer)
    image_filename = db.Column(db.String(200))
    owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)


class CaretakerProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    services = db.Column(db.String(500), default='[]')  # Stored as JSON string
    price_per_hour = db.Column(db.Float, nullable=False)
    description = db.Column(db.String(500))
    rating = db.Column(db.Float, default=0.0)
    service_location = db.Column(db.String(100), nullable=False)

    # Convert services to/from JSON
    def get_services(self):
        try:
            return json.loads(self.services)
        except:
            return []

    def set_services(self, services_list):
        self.services = json.dumps(services_list)


class Booking(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pet_owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    caretaker_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='pending')  # 'pending', 'confirmed', 'completed', 'cancelled'
    pet_id = db.Column(db.Integer, db.ForeignKey('pet.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Define relationships
    pet_owner = db.relationship('User', foreign_keys=[pet_owner_id])
    caretaker = db.relationship('User', foreign_keys=[caretaker_id])
    pet = db.relationship('Pet')
    review = db.relationship('Review', backref='booking', uselist=False)


class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    booking_id = db.Column(db.Integer, db.ForeignKey('booking.id'), nullable=True)
    service_request_id = db.Column(db.Integer, db.ForeignKey('pet_service_request.id'), nullable=True)
    rating = db.Column(db.Integer, nullable=False)  # 1-5 stars
    comment = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)



class ReadNotification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    notification_type = db.Column(db.String(50), nullable=False)  # 'service_accepted' or 'service_request'
    notification_id = db.Column(db.Integer, nullable=False)  # ID of the related entity (e.g., service request ID)
    read_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define a unique constraint to prevent duplicate read records
    __table_args__ = (db.UniqueConstraint('user_id', 'notification_type', 'notification_id', name='unique_read_notification'),)


class PetServiceRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pet_owner_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    pet_id = db.Column(db.Integer, db.ForeignKey('pet.id'), nullable=False)
    services_needed = db.Column(db.String(500))  # Stored as JSON string
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), default='open')  # 'open', 'accepted', 'completed', 'cancelled'
    caretaker_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(500))

    # Define relationships
    pet_owner = db.relationship('User', foreign_keys=[pet_owner_id])
    caretaker = db.relationship('User', foreign_keys=[caretaker_id])
    pet = db.relationship('Pet')
    review = db.relationship('Review', backref='service_request', uselist=False)

    def get_services_needed(self):
        if not self.services_needed:
            return []
        
        try:
            # Try to parse as JSON
            return json.loads(self.services_needed)
        except json.JSONDecodeError:
            # If not valid JSON, treat as comma-separated string
            return [service.strip() for service in self.services_needed.split(',') if service.strip()]

    def set_services_needed(self, services_list):
        self.services_needed = json.dumps(services_list)


# Breed information database


# Breed information database
breed_info = {
    # Dog breeds
    'american': {
        'diet': 'High-quality dog food with protein content of 25-30%',
        'exercise': '1-2 hours of exercise daily, including walks and play',
        'grooming': 'Brush 2-3 times a week, bathe every 6-8 weeks',
        'health': 'Regular check-ups, hip dysplasia monitoring'
    },
    'basset': {
        'diet': 'Measured portions, prone to obesity',
        'exercise': 'Moderate daily walks, scent-tracking activities',
        'grooming': 'Regular ear cleaning, weekly brushing',
        'health': 'Monitor for ear infections, weight management'
    },
    'beagle': {
        'diet': 'Measured portions to prevent obesity, 1-1.5 cups daily',
        'exercise': '1 hour of vigorous exercise daily, love to track scents',
        'grooming': 'Weekly brushing, occasional baths',
        'health': 'Weight management, ear infection prevention'
    },
    'boxer': {
        'diet': 'High-quality protein-rich food, 2-3 cups daily divided into two meals',
        'exercise': '1-2 hours of vigorous exercise daily, enjoys running and playtime',
        'grooming': 'Weekly brushing, occasional baths, clean facial wrinkles regularly',
        'health': 'Monitor for heart conditions, hip dysplasia, and cancer'
    },
    'chihuahua': {
        'diet': 'High-quality small breed formula, 1/4 to 1/2 cup daily',
        'exercise': '30 minutes of light exercise daily, short walks and indoor play',
        'grooming': 'Weekly brushing, dental care crucial, bathe monthly',
        'health': 'Dental problems, patellar luxation, hypoglycemia in puppies'
    },
    'english': {
        'diet': 'High-quality food formulated for large breeds, monitored portions',
        'exercise': '1 hour of moderate exercise daily, avoid strenuous activity in heat',
        'grooming': 'Regular cleaning of skin folds, weekly brushing, frequent drool management',
        'health': 'Monitor for hip dysplasia, heart issues, and skin infections'
    },
    'german': {
        'diet': 'High-protein diet, 3-4 cups of quality food daily',
        'exercise': '2+ hours of physical and mental exercise daily',
        'grooming': 'Brush 2-3 times weekly, seasonal heavy shedding',
        'health': 'Hip/elbow dysplasia screening, regular checkups'
    },
    'great': {
        'diet': 'Large breed formula, 6-10 cups daily divided into two meals',
        'exercise': 'Moderate daily exercise, 1-2 walks and play sessions',
        'grooming': 'Weekly brushing, clean facial wrinkles, manage drooling',
        'health': 'Monitor for bloat, hip dysplasia, and heart conditions'
    },
    'havanese': {
        'diet': 'High-quality small breed formula, 1/2 to 1 cup daily',
        'exercise': '30-60 minutes of play and short walks daily',
        'grooming': 'Daily brushing, regular professional grooming every 6-8 weeks',
        'health': 'Dental care, patellar luxation, eye conditions'
    },
    'japanese': {
        'diet': 'High-quality small breed formula, carefully measured portions',
        'exercise': '30-45 minutes of daily walks and play sessions',
        'grooming': 'Weekly brushing, regular ear cleaning, occasional baths',
        'health': 'Regular dental care, patellar luxation screening'
    },
    'keeshond': {
        'diet': 'High-quality food, 1-2 cups daily based on activity level',
        'exercise': '1 hour of moderate exercise daily, enjoys play and short walks',
        'grooming': 'Brush 2-3 times weekly, more during seasonal shedding',
        'health': 'Monitor for hip dysplasia, eye problems, and skin conditions'
    },
    'leonberger': {
        'diet': 'Large breed formula, 4-8 cups daily divided into two meals',
        'exercise': '1-2 hours of daily exercise, swimming and hiking ideal',
        'grooming': 'Thorough brushing 2-3 times weekly, more during shedding seasons',
        'health': 'Monitor for hip dysplasia, heart conditions, and bloat'
    },
    'miniature': {
        'diet': 'High-quality small breed formula, 1/2 to 1 cup daily',
        'exercise': '30-60 minutes of daily activity, mental stimulation important',
        'grooming': 'Professional grooming every 4-6 weeks, daily brushing',
        'health': 'Dental care, patellar luxation, progressive retinal atrophy'
    },
    'newfoundland': {
        'diet': 'Large breed formula, 4-5 cups daily divided into two meals',
        'exercise': '30-60 minutes of moderate exercise, loves swimming',
        'grooming': 'Thorough brushing 2-3 times weekly, manage heavy seasonal shedding',
        'health': 'Monitor for hip dysplasia, heart conditions, and bloat'
    },
    'pomeranian': {
        'diet': 'High-quality small breed formula, 1/4 to 1/2 cup daily',
        'exercise': '30 minutes of light exercise, short walks and play sessions',
        'grooming': 'Brush 2-3 times weekly, professional grooming every 6-8 weeks',
        'health': 'Dental care, patellar luxation, tracheal collapse'
    },
    'pug': {
        'diet': 'Measured portions to prevent obesity, 1/2 to 1 cup daily',
        'exercise': '30 minutes of light exercise, avoid exertion in hot weather',
        'grooming': 'Weekly brushing, clean facial wrinkles daily, monitor breathing',
        'health': 'Breathing issues, eye problems, obesity prevention'
    },
    'saint': {
        'diet': 'Large breed formula, 4-8 cups daily divided into two meals',
        'exercise': '30-60 minutes of moderate exercise, avoid strenuous activity',
        'grooming': 'Brush 2-3 times weekly, manage seasonal shedding and drooling',
        'health': 'Monitor for hip dysplasia, bloat, and heart conditions'
    },
    'samoyed': {
        'diet': 'High-quality protein-rich food, 2-3 cups daily',
        'exercise': '1-2 hours of daily exercise, enjoys cold weather activities',
        'grooming': 'Thorough brushing 2-3 times weekly, more during shedding season',
        'health': 'Monitor for hip dysplasia, eye problems, and diabetes'
    },
    'scottish': {
        'diet': 'High-quality terrier formula, 1-2 cups daily based on activity',
        'exercise': '1 hour of daily exercise, enjoys walks and playtime',
        'grooming': 'Professional grooming every 6-8 weeks, regular coat stripping',
        'health': "Dental care, von Willebrand's disease screening"
    },
    'shiba': {
        'diet': 'High-quality food, 1-1.5 cups daily based on activity level',
        'exercise': '1 hour of daily exercise, secure fencing required (escape artists)',
        'grooming': 'Weekly brushing, more during seasonal shedding',
        'health': 'Monitor for allergies, hip dysplasia, and eye problems'
    },
    'staffordshire': {
        'diet': 'High-quality protein-rich food, 2-3 cups daily',
        'exercise': '1-2 hours of vigorous exercise daily, enjoys strength activities',
        'grooming': 'Weekly brushing, occasional baths',
        'health': 'Monitor for hip dysplasia, skin allergies, and heart conditions'
    },
    'wheaten': {
        'diet': 'High-quality food, 1.5-2 cups daily divided into two meals',
        'exercise': '1 hour of daily exercise, enjoys walks and play sessions',
        'grooming': 'Daily brushing, professional grooming every 6-8 weeks',
        'health': 'Monitor for protein-losing nephropathy, protein-losing enteropathy'
    },
    'yorkshire': {
        'diet': 'High-quality small breed formula, 1/4 to 1/2 cup daily',
        'exercise': '30 minutes of light exercise, short walks and indoor play',
        'grooming': 'Daily brushing, professional grooming every 4-6 weeks',
        'health': 'Dental care, patellar luxation, tracheal collapse'
    },

    # Cat breeds
    'Abyssinian': {
        'diet': 'High-protein cat food, 1/4 to 1/2 cup daily',
        'exercise': 'Interactive toys, climbing trees, 20-30 minutes play daily',
        'grooming': 'Weekly brushing, minimal shedding',
        'health': 'Dental care, annual check-ups'
    },
    'Bengal': {
        'diet': 'High-quality protein-rich diet',
        'exercise': 'Highly active, needs climbing areas and interactive play',
        'grooming': 'Weekly brushing, occasional bath',
        'health': 'Monitor for heart issues, regular check-ups'
    },
    'Birman': {
        'diet': 'Premium quality cat food, 1/4 to 1/2 cup daily',
        'exercise': 'Moderate activity level, enjoys interactive toys and playtime',
        'grooming': 'Brush 2-3 times weekly to prevent matting, especially around collar',
        'health': 'Monitor for hypertrophic cardiomyopathy, kidney issues'
    },
    'Bombay': {
        'diet': 'High-quality cat food, measured portions to prevent obesity',
        'exercise': 'Interactive play sessions, enjoys running and jumping activities',
        'grooming': 'Weekly brushing, occasional wipe-down with pet-safe wipes',
        'health': 'Monitor for respiratory issues, regular dental check-ups'
    },
    'British': {
        'diet': 'Portion-controlled high-quality food, prone to obesity',
        'exercise': 'Daily play sessions, interactive toys to encourage movement',
        'grooming': 'Weekly brushing, more during seasonal shedding',
        'health': 'Monitor for hypertrophic cardiomyopathy, polycystic kidney disease'
    },
    'Egyptian': {
        'diet': 'High-protein diet, warm food preferred (room temperature)',
        'exercise': 'Very active breed, needs vertical spaces and interactive play',
        'grooming': 'Minimal grooming, occasional wiping with warm damp cloth',
        'health': 'Sensitive to cold, monitor for heart issues and respiratory problems'
    },
    'Maine': {
        'diet': 'High-quality cat food, 1/3 to 1/2 cup per 5 pounds of body weight daily',
        'exercise': 'Moderate activity level, enjoys interactive toys and climbing',
        'grooming': 'Brush 2-3 times weekly, more during seasonal shedding',
        'health': 'Monitor for hypertrophic cardiomyopathy, hip dysplasia'
    },
    'Persian': {
        'diet': 'Premium cat food, may need food for hairball control',
        'exercise': 'Moderate play, less active than other breeds',
        'grooming': 'Daily brushing, regular eye cleaning, monthly baths',
        'health': 'Polycystic kidney disease screening, breathing monitoring'
    },
    'Ragdoll': {
        'diet': 'High-quality cat food, monitor portions to prevent obesity',
        'exercise': 'Moderate activity level, enjoys interactive play',
        'grooming': 'Brush 2-3 times weekly to prevent matting',
        'health': 'Monitor for hypertrophic cardiomyopathy, urinary tract issues'
    },
    'Russian': {
        'diet': 'High-quality protein-rich diet, monitor for food allergies',
        'exercise': 'Active and playful, enjoys interactive toys and puzzle feeders',
        'grooming': 'Weekly brushing, minimal shedding',
        'health': 'Monitor for bladder stones, hypertrophic cardiomyopathy'
    },
    'Siamese': {
        'diet': 'High-quality cat food, measured portions to maintain lean build',
        'exercise': 'Very active breed, needs daily play and mental stimulation',
        'grooming': 'Weekly brushing, dental care important',
        'health': 'Monitor for respiratory issues, amyloidosis, dental problems'
    },
    'Sphynx': {
        'diet': 'High-calorie diet, needs more food than other breeds due to higher metabolism',
        'exercise': 'Active and energetic, needs interactive play',
        'grooming': 'Weekly baths, clean skin folds regularly, protect from sun exposure',
        'health': 'Monitor for hypertrophic cardiomyopathy, skin conditions'
    }
}


# Helper function to update expired bookings
def update_expired_bookings():
    """
    Update the status of bookings that have passed their end time to 'completed'
    if they were in 'confirmed' status.
    """
    now = datetime.utcnow()
    expired_bookings = Booking.query.filter(
        Booking.end_time < now,
        Booking.status == 'confirmed'
    ).all()

    for booking in expired_bookings:
        booking.status = 'completed'

    if expired_bookings:
        db.session.commit()
        print(f"Updated {len(expired_bookings)} bookings to 'completed' status.")


# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image at {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (250, 250))
        img = img.astype(np.float32) / 255.0  # Normalize to 0-1
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None


def predict_breed(image_path, animal_type):
    img = preprocess_image(image_path)
    if img is None:
        return None

    try:
        if animal_type == 'dog':
            if dog_model is None:
                flash("Dog breed identification is temporarily unavailable.", "warning")
                return None

            predictions = dog_model.predict(img)
            class_idx = np.argmax(predictions, axis=1)[0]

            # Dog breed classes (order matters!)
            dog_classes = [
                "american", "basset", "beagle", "boxer", "chihuahua",
                "english", "german", "great", "havanese", "japanese",
                "keeshond", "leonberger", "miniature", "newfoundland",
                "pomeranian", "pug", "saint", "samoyed", "scottish",
                "shiba", "staffordshire", "wheaten", "yorkshire"
            ]

            return dog_classes[class_idx]

        elif animal_type == 'cat':
            if cat_model is None:
                flash("Cat breed identification is temporarily unavailable.", "warning")
                return None

            predictions = cat_model.predict(img)
            class_idx = np.argmax(predictions, axis=1)[0]

            # Cat breed classes (order matters!)
            cat_classes = [
                "Abyssinian", "Bengal", "Birman", "Bombay", "British",
                "Egyptian", "Maine", "Persian", "Ragdoll", "Russian",
                "Siamese", "Sphynx"
            ]

            return cat_classes[class_idx]
    except Exception as e:
        print(f"Error in predict_breed: {e}")
        flash(f"An error occurred during breed prediction: {str(e)}", "error")

    return None


# Routes
@app.route('/')
def index():
    # Get notification data
    notifications_data = get_notifications()
    return render_template('index.html', 
                          unread_notifications=notifications_data['unread_notifications'],
                          all_notifications=notifications_data['all_notifications'])


@app.route('/debug-index')
def debug_index():
    """Debug version of the index route that shows detailed errors"""
    try:
        print("Beginning debug-index route")
        print(f"Models loaded: Dog: {dog_model is not None}, Cat: {cat_model is not None}")

        # Test if we can create a simple response
        response = "Debug started"
        print(response)

        # Try accessing the templates directory
        template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
        print(f"Template directory: {template_dir}")
        print(f"Template directory exists: {os.path.exists(template_dir)}")

        if os.path.exists(template_dir):
            files = os.listdir(template_dir)
            print(f"Files in template directory: {files}")

        # Try rendering a simple string template first
        from flask import render_template_string
        string_html = render_template_string("<html><body><h1>Test</h1></body></html>")
        print("String template rendered successfully")

        # Now try the actual template
        print("Attempting to render index.html")
        result = render_template('index.html')
        print("Successfully rendered index.html")

        return result
    except Exception as e:
        import traceback
        error_msg = f"<h1>Error rendering index.html</h1><pre>{str(e)}\n\n{traceback.format_exc()}</pre>"
        print(f"Error in debug-index: {str(e)}")
        return error_msg


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        user_type = request.form.get('user_type')
        phone = request.form.get('phone')
        alternate_phone = request.form.get('alternate_phone')
        address = request.form.get('address')
        city = request.form.get('city')
        state = request.form.get('state')
        pincode = request.form.get('pincode')
        
        # Validate form data
        if not all([name, email, password, confirm_password, user_type, phone, address, city, state, pincode]):
            flash('Please fill in all required fields.', 'danger')
            return redirect(url_for('register'))
        
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return redirect(url_for('register'))
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            user_type=user_type,
            phone=phone,
            alternate_phone=alternate_phone,
            address=address,
            city=city,
            state=state,
            pincode=pincode
        )
        
        try:
            db.session.add(new_user)
            db.session.commit()
            
            # If user is a caretaker, create their profile
            if user_type == 'caretaker':
                services = request.form.getlist('services')
                price_per_hour = request.form.get('price_per_hour')
                description = request.form.get('description', '')
                service_location = request.form.get('location')
                
                # Validate required caretaker fields
                if not all([price_per_hour, service_location]):
                    flash('Please fill in all required caretaker fields.', 'danger')
                    db.session.rollback()
                    return redirect(url_for('register'))
                
                new_profile = CaretakerProfile(
                    user_id=new_user.id,
                    services=json.dumps(services),  # Convert list to JSON string
                    price_per_hour=float(price_per_hour),
                    description=description,
                    service_location=service_location
                )
                db.session.add(new_profile)
                db.session.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred during registration: {str(e)}', 'danger')
            return redirect(url_for('register'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid email or password', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Get notification data
    notifications_data = get_notifications()
    if current_user.user_type == 'caretaker':
        profile = CaretakerProfile.query.filter_by(user_id=current_user.id).first()
        
        # Get recent bookings for display
        recent_bookings = Booking.query.filter_by(caretaker_id=current_user.id).order_by(Booking.start_time.desc()).limit(5).all()
        
        # Get accepted service requests from pet owners
        accepted_requests = PetServiceRequest.query.filter_by(
            caretaker_id=current_user.id,
            status='accepted'
        ).order_by(PetServiceRequest.start_time.desc()).all()
        
        # Get open service requests from pet owners
        open_requests = PetServiceRequest.query.filter_by(status='open').all()
        
        # Get total bookings count for accurate stats (including both regular bookings and service requests)
        booking_count = Booking.query.filter_by(caretaker_id=current_user.id).count()
        service_request_count = PetServiceRequest.query.filter_by(caretaker_id=current_user.id).count()
        total_bookings_count = booking_count + service_request_count
        
        # Get reviews from bookings only for now (until database is migrated)
        try:
            # Try to get reviews from both bookings and service requests
            booking_reviews = Review.query.join(Booking).filter(Booking.caretaker_id == current_user.id).all()
            
            # Check if service_request_id column exists
            service_reviews = []
            try:
                service_reviews = Review.query.join(PetServiceRequest).filter(PetServiceRequest.caretaker_id == current_user.id).all()
            except Exception as e:
                # If error occurs, service_request_id column might not exist yet
                app.logger.error(f"Error fetching service reviews: {str(e)}")
                service_reviews = []
            
            # Combine and sort all reviews by creation date (newest first)
            all_reviews = booking_reviews + service_reviews
        except Exception as e:
            # Fallback to just using booking reviews if there's an error
            app.logger.error(f"Error fetching reviews: {str(e)}")
            booking_reviews = Review.query.join(Booking).filter(Booking.caretaker_id == current_user.id).all()
            all_reviews = booking_reviews
        
        # Sort reviews by creation date
        all_reviews.sort(key=lambda x: x.created_at, reverse=True)
        recent_reviews = all_reviews[:5]  # Get only the 5 most recent reviews
        
        # Get total reviews count
        total_reviews_count = len(all_reviews)
        
        # Calculate average rating from all reviews
        average_rating = 0
        if total_reviews_count > 0:
            total_rating = sum(review.rating for review in all_reviews)
            average_rating = total_rating / total_reviews_count
        
        return render_template('caretaker_dashboard.html', 
                              profile=profile,
                              recent_bookings=recent_bookings,
                              accepted_requests=accepted_requests,
                              open_requests=open_requests,
                              recent_reviews=recent_reviews,
                              total_bookings_count=total_bookings_count,
                              total_reviews_count=total_reviews_count,
                              average_rating=average_rating,
                              unread_notifications=notifications_data['unread_notifications'],
                              all_notifications=notifications_data['all_notifications'])
    else:
        # Get all pets for the current user
        pets = Pet.query.filter_by(owner_id=current_user.id).all()
        app.logger.info(f"User {current_user.id} ({current_user.name}) has {len(pets)} pets")
        for pet in pets:
            app.logger.info(f"Pet: {pet.id}, {pet.name}, {pet.species}, {pet.breed}, {pet.owner_id}")
        
        # Get service requests for the pet owner
        service_requests = PetServiceRequest.query.filter_by(pet_owner_id=current_user.id).order_by(PetServiceRequest.created_at.desc()).all()
        app.logger.info(f"User {current_user.id} has {len(service_requests)} service requests")
        
        return render_template('owner_dashboard.html', 
                             pets=pets, 
                             service_requests=service_requests,
                             unread_notifications=notifications_data['unread_notifications'],
                             all_notifications=notifications_data['all_notifications'])


@app.route('/booking/accept/<int:booking_id>', methods=['POST'])
@login_required
def accept_booking(booking_id):
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can accept bookings', 'warning')
        return redirect(url_for('dashboard'))

    booking = Booking.query.get_or_404(booking_id)

    # Verify this caretaker is assigned to this booking
    if booking.caretaker_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    # Update booking status
    booking.status = 'confirmed'
    db.session.commit()
    
    # Get pet and owner information
    pet = Pet.query.get(booking.pet_id)
    pet_owner = User.query.get(booking.pet_owner_id)
    
    # Format dates for notification
    start_time = booking.start_time.strftime('%Y-%m-%d %H:%M')
    end_time = booking.end_time.strftime('%Y-%m-%d %H:%M')
    
    # Create notification for pet owner
    notification_message = f"Your booking request for {pet.name} has been accepted by {current_user.name} for {start_time} to {end_time}."
    create_notification(booking.pet_owner_id, notification_message)

    flash('Booking confirmed successfully!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/booking/decline/<int:booking_id>', methods=['POST'])
@login_required
def decline_booking(booking_id):
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can decline bookings', 'warning')
        return redirect(url_for('dashboard'))
    booking = Booking.query.get_or_404(booking_id)

    # Verify this caretaker is assigned to this booking
    if booking.caretaker_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    # Update booking status
    booking.status = 'declined'
    db.session.commit()
    
    # Get pet and owner information
    pet = Pet.query.get(booking.pet_id)
    pet_owner = User.query.get(booking.pet_owner_id)
    
    # Format dates for notification
    start_time = booking.start_time.strftime('%Y-%m-%d %H:%M')
    end_time = booking.end_time.strftime('%Y-%m-%d %H:%M')
    
    # Create notification for pet owner
    notification_message = f"Your booking request for {pet.name} has been declined by {current_user.name} for {start_time} to {end_time}."
    create_notification(booking.pet_owner_id, notification_message)

    flash('Booking declined', 'info')
    return redirect(url_for('dashboard'))


@app.route('/booking/complete/<int:booking_id>', methods=['POST'])
@login_required
def complete_booking(booking_id):
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can mark bookings as completed', 'warning')
        return redirect(url_for('dashboard'))
        
    booking = Booking.query.get_or_404(booking_id)

    # Verify this caretaker is assigned to this booking
    if booking.caretaker_id != current_user.id:
        flash('Unauthorized action', 'danger')
        return redirect(url_for('dashboard'))
        
    # Verify booking is in confirmed status
    if booking.status != 'confirmed':
        flash('Only confirmed bookings can be marked as completed', 'warning')
        return redirect(url_for('my_bookings'))

    # Update booking status
    booking.status = 'completed'
    db.session.commit()
    
    # Get pet and owner information
    pet = Pet.query.get(booking.pet_id)
    pet_owner = User.query.get(booking.pet_owner_id)
    
    # Format dates for notification
    start_time = booking.start_time.strftime('%Y-%m-%d %H:%M')
    end_time = booking.end_time.strftime('%Y-%m-%d %H:%M')
    
    # Create notification for pet owner
    notification_message = f"Your booking for {pet.name} has been marked as completed by {current_user.name}."
    create_notification(booking.pet_owner_id, notification_message)

    flash('Service has been marked as completed!', 'success')
    return redirect(url_for('my_bookings'))


@app.route('/booking/owner/cancel/<int:booking_id>', methods=['POST'])
@login_required
def owner_cancel_booking(booking_id):
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can cancel their bookings', 'warning')
        return redirect(url_for('dashboard'))

    booking = Booking.query.get_or_404(booking_id)

    # Verify this pet owner owns this booking
    if booking.pet_owner_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    # Update booking status
    booking.status = 'cancelled'
    db.session.commit()
    
    # Get pet and caretaker information
    pet = Pet.query.get(booking.pet_id)
    caretaker = User.query.get(booking.caretaker_id)
    
    # Format dates for notification
    start_time = booking.start_time.strftime('%Y-%m-%d %H:%M')
    end_time = booking.end_time.strftime('%Y-%m-%d %H:%M')
    
    # Create notification for caretaker
    notification_message = f"Booking for {pet.name} from {start_time} to {end_time} has been cancelled by the pet owner {current_user.name}."
    create_notification(booking.caretaker_id, notification_message)

    flash('Booking has been cancelled.', 'info')
    return redirect(url_for('dashboard'))


@app.route('/diet-recommendation', methods=['GET', 'POST'])
@login_required
def diet_recommendation():
    if request.method == 'POST':
        # Check if this is a detailed diet plan request (from identification_result.html)
        detailed_diet = request.form.get('detailed_diet', 'no')
        
        # Handle file upload or existing file
        if 'pet_image' in request.files and request.files['pet_image'].filename != '':
            # New file upload
            file = request.files['pet_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
            else:
                flash('Invalid file type', 'danger')
                return redirect(request.url)
        elif 'pet_image' in request.form and request.form['pet_image'] != '':
            # Using existing file from previous identification
            filename = request.form['pet_image']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(file_path):
                flash('File not found', 'danger')
                return redirect(request.url)
        else:
            flash('No file selected', 'warning')
            return redirect(request.url)

        # Now we have a valid file_path and filename
        # Get animal type from form
        animal_type = request.form.get('animal_type')

        # Check if model is available
        if (animal_type == 'dog' and dog_model is None) or (animal_type == 'cat' and cat_model is None):
            flash(f"Sorry, {animal_type} breed identification is temporarily unavailable.", "warning")

            # Still save the pet if name is provided
            pet_name = request.form.get('pet_name', '')
            if pet_name:
                new_pet = Pet(
                    name=pet_name,
                    species=animal_type,
                    breed="Unknown",  # Set to unknown since we can't identify
                    age=request.form.get('pet_age', 0),
                    image_filename=filename,
                    owner_id=current_user.id
                )
                db.session.add(new_pet)
                db.session.commit()
                flash('Pet added to your profile, but breed could not be identified.')

            return redirect(url_for('dashboard'))

        # Predict breed
        breed = predict_breed(file_path, animal_type)

        if breed:
            # Get breed information
            info = breed_info.get(breed, {
                'diet': 'Information not available',
                'exercise': 'Information not available',
                'grooming': 'Information not available',
                'health': 'Information not available'
            })
            
            # Create detailed diet information based on breed
            diet_details = generate_detailed_diet_plan(breed, animal_type)

            # If the user wants to save this pet
            pet_name = request.form.get('pet_name', '')
            if pet_name:
                new_pet = Pet(
                    name=pet_name,
                    species=animal_type,
                    breed=breed,
                    age=request.form.get('pet_age', 0),
                    image_filename=filename,
                    owner_id=current_user.id
                )
                db.session.add(new_pet)
                db.session.commit()
                flash('Pet added to your profile!')

            # Check if the user wants a detailed diet plan
            detailed_diet = request.form.get('detailed_diet', 'no')
            if detailed_diet == 'yes':
                return render_template('detailed_diet_plan.html',
                                   breed=breed,
                                   animal_type=animal_type,
                                   info=info,
                                   diet_details=diet_details,
                                   image_path=url_for('static', filename=f'uploads/{filename}'))
            else:
                return render_template('identification_result.html',
                                   breed=breed,
                                   animal_type=animal_type,
                                   info=info,
                                   image_path=url_for('static', filename=f'uploads/{filename}'))
        else:
            flash('Unable to identify breed. Please try again.')
            return redirect(request.url)

    # Check if models are available and set warning messages
    model_status = {
        'dog': dog_model is not None,
        'cat': cat_model is not None
    }

    return render_template('identify.html', model_status=model_status)


def generate_detailed_diet_plan(breed, animal_type):
    """Generate a detailed diet plan based on the breed and animal type."""
    # Base diet details template
    diet_details = {
        'nutritional_needs': '',
        'caloric_requirements': '',
        'feeding_schedule': '',
        'special_considerations': '',
        'recommended_foods': [],
        'foods_to_avoid': ['Chocolate', 'Grapes/raisins', 'Onions/garlic', 'Xylitol', 'Alcohol', 'Caffeine', 'Raw dough'],
        'supplements': '',
        'treats': '',
        'sample_meal_plan': {
            'day1': {'morning': '', 'afternoon': '', 'evening': ''},
            'day2': {'morning': '', 'afternoon': '', 'evening': ''},
            'day3': {'morning': '', 'afternoon': '', 'evening': ''},
            'day4': {'morning': '', 'afternoon': '', 'evening': ''}
        }
    }
    
    # Get basic diet info from breed_info
    basic_diet = breed_info.get(breed, {}).get('diet', 'Information not available')
    
    if animal_type == 'dog':
        # Dog-specific diet details
        if 'small' in breed.lower() or breed.lower() in ['yorkshire', 'chihuahua', 'pomeranian', 'havanese', 'japanese']:
            # Small breed dogs
            diet_details['nutritional_needs'] = 'Small breeds have higher metabolic rates and require more calories per pound than larger breeds. They need high-quality protein, healthy fats, and complex carbohydrates.'
            diet_details['caloric_requirements'] = 'Approximately 40 calories per pound of body weight daily. A typical small breed dog weighing 10 pounds needs about 400 calories per day.'
            diet_details['feeding_schedule'] = '3-4 small meals per day is recommended for small breeds to maintain stable blood sugar levels and prevent hypoglycemia.'
            diet_details['special_considerations'] = 'Small breeds are prone to dental issues, so consider dental health formulas. They also benefit from small kibble size designed for their smaller mouths.'
            diet_details['recommended_foods'] = ['High-quality small breed formula dog food', 'Lean protein sources (chicken, turkey, fish)', 'Complex carbohydrates (brown rice, sweet potatoes)', 'Vegetables (carrots, green beans, peas)', 'Healthy fats (fish oil, flaxseed)']
            diet_details['supplements'] = 'Consider dental supplements, joint support (especially for breeds prone to patellar luxation), and omega-3 fatty acids for coat health.'
            diet_details['treats'] = 'Small, low-calorie treats are best. Dental chews can help maintain oral health. Limit treats to 10% of daily caloric intake.'
            
        elif 'large' in breed.lower() or breed.lower() in ['german', 'great', 'saint', 'newfoundland', 'leonberger']:
            # Large breed dogs
            diet_details['nutritional_needs'] = 'Large breeds need controlled protein and calcium levels to support proper growth without causing skeletal issues. They require foods that support joint health and prevent bloat.'
            diet_details['caloric_requirements'] = 'Approximately 20-30 calories per pound of body weight daily. A typical large breed dog weighing 70 pounds needs about 1400-2100 calories per day.'
            diet_details['feeding_schedule'] = '2 meals per day, served in elevated bowls to reduce the risk of bloat. Avoid exercise for 1 hour before and after meals.'
            diet_details['special_considerations'] = 'Large breeds are prone to joint issues and bloat. Consider foods with glucosamine and chondroitin for joint health. Avoid foods that cause rapid growth in puppies.'
            diet_details['recommended_foods'] = ['Large breed specific formula dog food', 'Moderate protein sources (chicken, beef, fish)', 'Controlled calcium content foods', 'Vegetables and fruits for antioxidants', 'Sources of glucosamine and chondroitin']
            diet_details['supplements'] = 'Joint supplements containing glucosamine, chondroitin, and MSM are beneficial. Consider probiotics for digestive health and fish oil for coat and joint support.'
            diet_details['treats'] = 'Large, durable chews that promote dental health. Avoid small treats that could be a choking hazard. Monitor caloric intake from treats carefully.'
            
        else:
            # Medium breed dogs
            diet_details['nutritional_needs'] = 'Medium breeds benefit from balanced nutrition with moderate protein and fat levels. They need a diet that maintains healthy weight and supports overall health.'
            diet_details['caloric_requirements'] = 'Approximately 30-35 calories per pound of body weight daily. A typical medium breed dog weighing 40 pounds needs about 1200-1400 calories per day.'
            diet_details['feeding_schedule'] = '2 meals per day, morning and evening, with consistent timing to establish a routine.'
            diet_details['special_considerations'] = 'Medium breeds can be prone to weight gain. Monitor food intake and adjust as needed based on activity level and body condition.'
            diet_details['recommended_foods'] = ['High-quality adult dog food', 'Lean protein sources (chicken, turkey, fish)', 'Whole grains (brown rice, barley)', 'Vegetables and fruits for vitamins and fiber', 'Healthy fats for coat health']
            diet_details['supplements'] = 'Multivitamins may be beneficial, along with omega-3 fatty acids for skin and coat health. Consider joint supplements for active dogs.'
            diet_details['treats'] = 'Medium-sized treats appropriate for the breed. Fresh vegetables like carrots and green beans make healthy, low-calorie treats. Limit treats to 10% of daily caloric intake.'
    
    elif animal_type == 'cat':
        # Cat-specific diet details
        if breed.lower() in ['persian', 'maine', 'british', 'ragdoll']:
            # Long-haired or large cats
            diet_details['nutritional_needs'] = 'Long-haired and larger cats need high-quality protein for muscle maintenance and special formulas that help with hairball control.'
            diet_details['caloric_requirements'] = 'Approximately 20 calories per pound of body weight daily. A typical large cat weighing 15 pounds needs about 300 calories per day.'
            diet_details['feeding_schedule'] = '2-3 meals per day, with fresh water always available. Consider puzzle feeders to slow eating and provide mental stimulation.'
            diet_details['special_considerations'] = 'Long-haired cats need diets that help with hairball control. Larger cats may be prone to joint issues and benefit from foods with joint support.'
            diet_details['recommended_foods'] = ['High-quality cat food with hairball control', 'High protein sources (chicken, turkey, fish)', 'Wet food to ensure adequate hydration', 'Limited carbohydrates', 'Foods with added omega fatty acids for coat health']
            diet_details['supplements'] = 'Omega-3 fatty acids for coat health, joint supplements for larger cats, and possibly digestive enzymes to help with hairball prevention.'
            diet_details['treats'] = 'Hairball control treats, freeze-dried meat treats, and occasional fresh boneless fish. Limit treats to prevent weight gain.'
            
        elif breed.lower() in ['bengal', 'abyssinian', 'siamese', 'egyptian']:
            # Active breeds
            diet_details['nutritional_needs'] = 'Active breeds require higher protein and calorie content to fuel their energy needs. They benefit from diets rich in animal protein and healthy fats.'
            diet_details['caloric_requirements'] = 'Approximately 25-30 calories per pound of body weight daily. A typical active cat weighing 10 pounds needs about 250-300 calories per day.'
            diet_details['feeding_schedule'] = 'Multiple small meals throughout the day or free feeding may work well for very active cats. Ensure fresh water is always available.'
            diet_details['special_considerations'] = 'Active breeds may need more calories than average. Monitor weight and adjust food intake accordingly. Consider high-protein diets to support muscle maintenance.'
            diet_details['recommended_foods'] = ['High-protein cat food formulated for active cats', 'Quality animal proteins (chicken, turkey, rabbit)', 'Wet food for hydration', 'Foods with higher fat content for energy', 'Limited carbohydrates']
            diet_details['supplements'] = 'B-vitamins for energy metabolism, omega fatty acids for overall health, and possibly L-carnitine for fat metabolism in very active cats.'
            diet_details['treats'] = 'High-protein treats like freeze-dried meat. Interactive treat toys can help satisfy hunting instincts while providing nutrition.'
            
        else:
            # Average cats
            diet_details['nutritional_needs'] = 'Cats are obligate carnivores and require high-quality animal protein. They need taurine, arachidonic acid, and vitamin A from animal sources.'
            diet_details['caloric_requirements'] = 'Approximately 20 calories per pound of body weight daily. A typical cat weighing 10 pounds needs about 200 calories per day.'
            diet_details['feeding_schedule'] = '2 meals per day, with consistent timing. Some cats do well with measured free feeding. Always provide fresh water.'
            diet_details['special_considerations'] = 'Indoor cats are prone to weight gain and may need calorie-controlled diets. Ensure adequate protein to maintain muscle mass.'
            diet_details['recommended_foods'] = ['High-quality cat food with animal protein as first ingredient', 'Mix of wet and dry food for balanced nutrition', 'Foods with minimal carbohydrates', 'Sources of taurine and essential fatty acids', 'Fresh water always available']
            diet_details['supplements'] = "Most cats on complete and balanced diets don't need supplements. Consider omega-3 fatty acids for skin and coat health if needed."
            diet_details['treats'] = 'Small portions of cooked meat, freeze-dried treats, or commercial cat treats. Limit to prevent weight gain and nutritional imbalances.'
    
    # Generate sample meal plan based on animal type and breed
    if animal_type == 'dog':
        diet_details['sample_meal_plan'] = {
            'day1': {
                'morning': f'1/2 cup high-quality {breed} formula dry food with 1 tablespoon plain yogurt',
                'afternoon': 'Small dental chew or 1/4 cup vegetables as a snack',
                'evening': f'1/2 cup high-quality {breed} formula dry food mixed with 1/4 cup wet food'
            },
            'day2': {
                'morning': f'1/2 cup high-quality {breed} formula dry food with 1 teaspoon fish oil',
                'afternoon': 'Treat-dispensing toy with a small portion of kibble',
                'evening': '3/4 cup lean protein (cooked chicken or turkey) with 1/4 cup cooked vegetables'
            },
            'day3': {
                'morning': f'1/2 cup high-quality {breed} formula dry food with a scrambled egg',
                'afternoon': 'Carrot sticks or apple slices as a healthy snack',
                'evening': f'1/2 cup high-quality {breed} formula dry food with 1/4 cup cottage cheese'
            },
            'day4': {
                'morning': f'1/2 cup high-quality {breed} formula dry food with 1 tablespoon pumpkin puree',
                'afternoon': 'Small dental chew or training treats during short training session',
                'evening': f'1/2 cup high-quality {breed} formula wet food with 1/4 cup brown rice'
            }
        }
    else:  # cat
        diet_details['sample_meal_plan'] = {
            'day1': {
                'morning': '1/4 cup high-quality dry cat food',
                'afternoon': 'Small portion of wet food (1-2 tablespoons) as a snack',
                'evening': '1/3 cup wet cat food with added water for hydration'
            },
            'day2': {
                'morning': '1/4 cup high-quality dry cat food with 1 teaspoon fish oil',
                'afternoon': 'Treat-dispensing toy with a few kibbles',
                'evening': 'Small portion of cooked chicken or fish (2 tablespoons) with wet food'
            },
            'day3': {
                'morning': '1/4 cup high-quality dry cat food',
                'afternoon': 'Freeze-dried meat treats during play session',
                'evening': '1/3 cup wet cat food mixed with a sprinkle of nutritional yeast'
            },
            'day4': {
                'morning': '1/4 cup high-quality dry cat food with probiotic sprinkled on top',
                'afternoon': 'Small amount of wet food (1-2 tablespoons)',
                'evening': '1/3 cup wet cat food with a small amount of pureed pumpkin for fiber'
            }
        }
    
    # Add foods to avoid based on animal type
    if animal_type == 'dog':
        diet_details['foods_to_avoid'].extend(['Macadamia nuts', 'Avocado', 'Cooked bones', 'Corn cobs', 'Artificial sweeteners'])
    else:  # cat
        diet_details['foods_to_avoid'].extend(['Dog food', 'Tuna as a staple diet', 'Raw fish', 'Dairy products', 'Liver in large amounts'])
    
    return diet_details


@app.route('/available-pets')
def available_pets():
    # Get all available pets
    pets = Pet.query.all()
    
    # Get pet owners for each pet
    pet_owners = {}
    for pet in pets:
        owner = User.query.get(pet.owner_id)
        if owner:
            pet_owners[pet.id] = owner.name
    
    # Get all open service requests
    open_requests = PetServiceRequest.query.filter_by(status='open').all()
    
    # Check if user is authenticated and is a pet owner
    is_pet_owner = current_user.is_authenticated and current_user.user_type == 'pet_owner'
    
    return render_template('available_pets.html',
                         pets=pets,
                         pet_owners=pet_owners,
                         open_requests=open_requests,
                         is_pet_owner=is_pet_owner)


@app.route('/book-pet/<int:pet_id>', methods=['GET', 'POST'])
@login_required
def book_pet(pet_id):
    # Check if user is a caretaker
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can book pets.', 'danger')
        return redirect(url_for('available_pets'))
    
    # Get the pet
    pet = Pet.query.get_or_404(pet_id)
    
    # Get the pet owner
    pet_owner = User.query.get(pet.owner_id)
    
    if request.method == 'POST':
        # Get form data
        start_time_str = request.form.get('start_time')
        end_time_str = request.form.get('end_time')
        services = request.form.getlist('services')
        description = request.form.get('description')
        
        # Validate form data
        if not start_time_str or not end_time_str or not services:
            flash('Please fill in all required fields.', 'danger')
            return render_template('book_pet.html', pet=pet, pet_owner=pet_owner)
            
        # Convert string datetime to Python datetime objects
        try:
            start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
            end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')
        except ValueError:
            flash('Invalid date format. Please use the date picker.', 'danger')
            return render_template('book_pet.html', pet=pet, pet_owner=pet_owner)
        
        # Create a new service request
        service_request = PetServiceRequest(
            pet_id=pet.id,
            pet_owner_id=pet.owner_id,
            caretaker_id=current_user.id,
            start_time=start_time,
            end_time=end_time,
            services_needed=json.dumps(services),
            status='pending',
            description=description
        )
        
        # Save the service request
        db.session.add(service_request)
        db.session.commit()
        
        flash('Your booking request has been sent to the pet owner.', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('book_pet.html', pet=pet, pet_owner=pet_owner)


@app.route('/caretakers')
def list_caretakers():
    # Get notification data
    notifications_data = get_notifications()
    
    location = request.args.get('location', '')

    # Query for caretakers, with optional location filter
    query = db.session.query(User, CaretakerProfile).join(CaretakerProfile)

    if location:
        query = query.filter(User.location == location)

    caretakers = query.all()
    
    # Get all available pets
    available_pets = Pet.query.all()
    
    # Get pet owners for each pet
    pet_owners = {}
    for pet in available_pets:
        owner = User.query.get(pet.owner_id)
        if owner:
            pet_owners[pet.id] = owner.name

    # Get all open service requests
    open_requests = PetServiceRequest.query.filter_by(status='open').all()

    # Check if user is authenticated and is a pet owner
    is_pet_owner = current_user.is_authenticated and current_user.user_type == 'pet_owner'

    return render_template('caretakers.html', 
                         caretakers=caretakers, 
                         filter_location=location,
                         is_pet_owner=is_pet_owner,
                         open_requests=open_requests,
                         available_pets=available_pets,
                         pet_owners=pet_owners,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])



@app.route('/book/<int:caretaker_id>', methods=['GET', 'POST'])
@login_required
def book_caretaker(caretaker_id):
    # Get notification data
    notifications_data = get_notifications()
    
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can book caretakers', 'warning')
        return redirect(url_for('list_caretakers'))

    caretaker = User.query.get_or_404(caretaker_id)

    if request.method == 'POST':
        pet_id = request.form.get('pet_id')
        start_time_str = request.form.get('start_time')
        end_time_str = request.form.get('end_time')

        # Convert string times to datetime objects
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')

        # Create booking
        booking = Booking(
            pet_owner_id=current_user.id,
            caretaker_id=caretaker_id,
            pet_id=pet_id,
            start_time=start_time,
            end_time=end_time
        )

        db.session.add(booking)
        db.session.commit()
        
        # Get pet information
        pet = Pet.query.get(pet_id)
        
        # Create notification for caretaker
        notification_message = f"{current_user.name} has requested you to take care of their pet {pet.name} from {start_time_str} to {end_time_str}."
        create_notification(caretaker_id, notification_message)

        flash('Booking request sent!', 'success')
        return redirect(url_for('dashboard'))

    # Get user's pets for the booking form
    pets = Pet.query.filter_by(owner_id=current_user.id).all()

    return render_template('book_caretaker.html', 
                         caretaker=caretaker, 
                         pets=pets,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


@app.route('/profile')
@login_required
def profile():
    # Get notification data
    notifications_data = get_notifications()
    if current_user.user_type == 'caretaker':
        # Get all bookings for this caretaker
        all_bookings = Booking.query.filter_by(caretaker_id=current_user.id).all()
        pending_bookings = [b for b in all_bookings if b.status == 'pending']
        confirmed_bookings = [b for b in all_bookings if b.status == 'confirmed' and b.end_time > datetime.utcnow()]
        completed_bookings = [b for b in all_bookings if b.status == 'completed']

        # Get all pets from active bookings
        active_pets = []
        for booking in pending_bookings + confirmed_bookings:
            if booking.pet not in active_pets:
                active_pets.append(booking.pet)

        # Calculate earnings
        now = datetime.utcnow()
        
        # Current month earnings
        month_start = datetime(now.year, now.month, 1)
        month_end = datetime(now.year + (1 if now.month == 12 else 0), 1 if now.month == 12 else now.month + 1, 1)
        month_earnings = sum(((b.end_time - b.start_time).total_seconds() / 3600) * current_user.caretaker_profile.price_per_hour
                           for b in completed_bookings if month_start <= b.start_time < month_end)
        month_earnings = round(month_earnings, 2)

        # Last month earnings
        last_month = now.month - 1 if now.month > 1 else 12
        last_month_year = now.year if now.month > 1 else now.year - 1
        last_month_start = datetime(last_month_year, last_month, 1)
        last_month_end = datetime(now.year, now.month, 1)
        last_month_earnings = sum(((b.end_time - b.start_time).total_seconds() / 3600) * current_user.caretaker_profile.price_per_hour
                                for b in completed_bookings if last_month_start <= b.start_time < last_month_end)
        last_month_earnings = round(last_month_earnings, 2)

        # Year earnings
        year_start = datetime(now.year, 1, 1)
        year_end = datetime(now.year + 1, 1, 1)
        earnings_year = sum(((b.end_time - b.start_time).total_seconds() / 3600) * current_user.caretaker_profile.price_per_hour
                          for b in completed_bookings if year_start <= b.start_time < year_end)
        earnings_year = round(earnings_year, 2)

        return render_template('caretaker_profile.html',
                             user=current_user,
                             profile=current_user.caretaker_profile,
                             all_bookings=all_bookings,
                             pending_bookings=pending_bookings,
                             confirmed_bookings=confirmed_bookings,
                             completed_bookings=completed_bookings,
                             month_earnings=month_earnings,
                             last_month_earnings=last_month_earnings,
                             earnings_year=earnings_year,
                             current_year=now.year,
                             active_pets=active_pets,
                             unread_notifications=notifications_data['unread_notifications'],
                             all_notifications=notifications_data['all_notifications'])
    else:
        return render_template('owner_profile.html', 
                             user=current_user,
                             unread_notifications=notifications_data['unread_notifications'],
                             all_notifications=notifications_data['all_notifications'])


@app.context_processor
def inject_year():
    return {'current_year': datetime.now().year}


@app.route('/add_pet', methods=['GET', 'POST'])
@login_required
def add_pet():
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can add pets', 'warning')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        name = request.form.get('name')
        species = request.form.get('species')
        breed = request.form.get('breed', '')
        age = request.form.get('age', 0)

        # Handle image upload
        image_filename = None
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_filename = filename

        # Create new pet
        new_pet = Pet(
            name=name,
            species=species,
            breed=breed,
            age=age,
            image_filename=image_filename,
            owner_id=current_user.id
        )

        db.session.add(new_pet)
        db.session.commit()

        flash('Pet added successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('add_pet.html')


@app.route('/update_caretaker_services', methods=['POST'])
@login_required
def update_caretaker_services():
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can update their services', 'error')
        return redirect(url_for('profile'))

    try:
        # Get form data
        price_per_hour = float(request.form.get('price_per_hour', 0))
        description = request.form.get('description', '')
        services = request.form.getlist('services')  # This gets all checked services

        # Update caretaker profile
        profile = current_user.caretaker_profile
        profile.price_per_hour = price_per_hour
        profile.description = description
        profile.set_services(services)

        db.session.commit()
        flash('Services updated successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error updating services: {str(e)}', 'error')

    return redirect(url_for('profile'))


@app.route('/create_service_request', methods=['GET', 'POST'])
@login_required
def create_service_request():
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can create service requests', 'warning')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        pet_id = request.form.get('pet_id')
        services = request.form.getlist('services')
        start_time_str = request.form.get('start_time')
        end_time_str = request.form.get('end_time')
        description = request.form.get('description', '')

        # Convert string times to datetime objects
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%dT%H:%M')

        # Create service request
        service_request = PetServiceRequest(
            pet_owner_id=current_user.id,
            pet_id=pet_id,
            start_time=start_time,
            end_time=end_time,
            description=description
        )
        service_request.set_services_needed(services)

        db.session.add(service_request)
        db.session.commit()

        flash('Service request created successfully!', 'success')
        return redirect(url_for('dashboard'))

    # Get user's pets for the form
    pets = Pet.query.filter_by(owner_id=current_user.id).all()
    return render_template('create_service_request.html', pets=pets)


@app.route('/service_requests')
@login_required
def list_service_requests():
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can view service requests', 'warning')
        return redirect(url_for('dashboard'))

    # Get all open service requests
    open_requests = PetServiceRequest.query.filter_by(status='open').all()
    # Get requests sent by this caretaker to pet owners
    my_requests = PetServiceRequest.query.filter_by(caretaker_id=current_user.id).all()

    return render_template('service_requests.html', 
                         open_requests=open_requests,
                         my_requests=my_requests)


@app.route('/service_request/<int:request_id>')
@login_required
def view_service_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is either the pet owner or the caretaker
    if current_user.id != service_request.pet_owner_id and current_user.id != service_request.caretaker_id:
        flash('You do not have permission to view this service request', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get pet owner and caretaker information
    pet_owner = User.query.get(service_request.pet_owner_id)
    
    caretaker = None
    caretaker_profile = None
    if service_request.caretaker_id:
        caretaker = User.query.get(service_request.caretaker_id)
        caretaker_profile = CaretakerProfile.query.filter_by(user_id=service_request.caretaker_id).first()
    
    # Get notifications for the current user
    notifications_data = get_notifications()
    
    return render_template('service_request_details.html',
                         request=service_request,
                         pet_owner=pet_owner,
                         caretaker=caretaker,
                         caretaker_profile=caretaker_profile,
                         notifications=notifications_data)


@app.route('/accept_service_request/<int:request_id>', methods=['POST'])
@login_required
def accept_service_request(request_id):
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can accept service requests')
        return redirect(url_for('dashboard'))

    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    if service_request.status != 'open':
        flash('This service request is no longer available')
        return redirect(url_for('list_service_requests'))

    # Update service request
    service_request.status = 'accepted'
    service_request.caretaker_id = current_user.id
    db.session.commit()

    flash('Service request accepted successfully! You can now view contact information.')
    return redirect(url_for('view_service_request', request_id=request_id))


@app.route('/complete_service_request/<int:request_id>', methods=['POST'])
@login_required
def complete_service_request(request_id):
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can complete service requests')
        return redirect(url_for('dashboard'))

    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    if service_request.caretaker_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    # Update service request
    service_request.status = 'completed'
    db.session.commit()
    
    # Get pet and owner information
    pet = Pet.query.get(service_request.pet_id)
    owner = User.query.get(service_request.owner_id)
    
    # Create notification for pet owner
    notification_message = f"Your pet {pet.name}'s care service has been marked as completed by {current_user.name}."
    create_notification(service_request.owner_id, notification_message)

    flash('Service request marked as completed!')
    return redirect(url_for('view_service_request', request_id=request_id))


@app.route('/submit_review/<int:request_id>', methods=['GET', 'POST'])
@login_required
def submit_review(request_id):
    # Get notification data
    notifications_data = get_notifications()
    
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can submit reviews')
        return redirect(url_for('dashboard'))

    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Verify this pet owner owns this service request
    if service_request.pet_owner_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))
    
    # Verify the service request is completed
    if service_request.status != 'completed':
        flash('You can only review completed services')
        return redirect(url_for('view_service_request', request_id=request_id))
    
    # Check if a review already exists for this service request
    existing_review = Review.query.filter_by(service_request_id=request_id).first()
    if existing_review:
        flash('You have already submitted a review for this service')
        return redirect(url_for('view_service_request', request_id=request_id))
    
    if request.method == 'POST':
        rating = request.form.get('rating')
        comment = request.form.get('comment')
        
        if not rating:
            flash('Please provide a rating')
            return redirect(url_for('submit_review', request_id=request_id))
        
        # Create a new review
        review = Review(
            service_request_id=request_id,
            rating=int(rating),
            comment=comment,
            created_at=datetime.utcnow()
        )
        db.session.add(review)
        
        # Update caretaker's average rating
        caretaker_profile = CaretakerProfile.query.filter_by(user_id=service_request.caretaker_id).first()
        if caretaker_profile:
            # Get all reviews for this caretaker
            caretaker_reviews = Review.query.join(PetServiceRequest).filter(
                PetServiceRequest.caretaker_id == service_request.caretaker_id
            ).all()
            
            # Calculate new average rating
            total_rating = sum(r.rating for r in caretaker_reviews) + int(rating)
            new_avg_rating = total_rating / (len(caretaker_reviews) + 1)
            
            # Update caretaker profile
            caretaker_profile.rating = new_avg_rating
        
        db.session.commit()
        
        # Create notification for caretaker
        pet = Pet.query.get(service_request.pet_id)
        notification_message = f"{current_user.name} has submitted a {rating}-star review for taking care of their pet {pet.name}."
        create_notification(service_request.caretaker_id, notification_message)
        
        flash('Thank you for your review!')
        return redirect(url_for('dashboard'))
    
    return render_template('submit_review.html', 
                         service_request=service_request,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


@app.route('/edit_pet/<int:pet_id>', methods=['GET', 'POST'])
@login_required
def edit_pet(pet_id):
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can edit pets')
        return redirect(url_for('dashboard'))

    pet = Pet.query.get_or_404(pet_id)
    
    # Verify the pet belongs to the current user
    if pet.owner_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        pet.name = request.form.get('name')
        pet.species = request.form.get('species')
        pet.breed = request.form.get('breed', '')
        pet.age = request.form.get('age', 0)

        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Delete old image if it exists
                if pet.image_filename:
                    old_file_path = os.path.join(app.config['UPLOAD_FOLDER'], pet.image_filename)
                    if os.path.exists(old_file_path):
                        os.remove(old_file_path)
                
                pet.image_filename = filename

        db.session.commit()
        flash('Pet updated successfully!')
        return redirect(url_for('dashboard'))

    return render_template('edit_pet.html', pet=pet)


@app.route('/delete_pet/<int:pet_id>', methods=['POST'])
@login_required
def delete_pet(pet_id):
    if current_user.user_type != 'pet_owner':
        flash('Only pet owners can delete pets')
        return redirect(url_for('dashboard'))

    pet = Pet.query.get_or_404(pet_id)
    
    # Verify the pet belongs to the current user
    if pet.owner_id != current_user.id:
        flash('Unauthorized action')
        return redirect(url_for('dashboard'))

    # Delete pet's image if it exists
    if pet.image_filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], pet.image_filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    db.session.delete(pet)
    db.session.commit()
    flash('Pet deleted successfully!')
    return redirect(url_for('dashboard'))


@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    # Get notification data
    notifications_data = get_notifications()
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can edit their profile')
        return redirect(url_for('dashboard'))

    profile = CaretakerProfile.query.filter_by(user_id=current_user.id).first()
    
    if request.method == 'POST':
        try:
            # Update user information
            current_user.name = request.form.get('name')
            current_user.location = request.form.get('location')
            current_user.phone = request.form.get('phone')
            current_user.alternate_phone = request.form.get('alternate_phone')
            current_user.address = request.form.get('address')
            current_user.city = request.form.get('city')
            current_user.state = request.form.get('state')
            current_user.pincode = request.form.get('pincode')

            # Update caretaker profile
            profile.price_per_hour = float(request.form.get('price_per_hour'))
            profile.description = request.form.get('description', '')
            profile.service_location = request.form.get('service_location')
            profile.set_services(request.form.getlist('services'))

            db.session.commit()
            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating profile: {str(e)}', 'danger')

    return render_template('edit_profile.html', 
                         user=current_user, 
                         profile=profile,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


@app.route('/my_bookings')
@login_required
def my_bookings():
    # Get notification data
    notifications_data = get_notifications()
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can view their bookings')
        return redirect(url_for('dashboard'))

    # Get all bookings for this caretaker
    bookings = Booking.query.filter_by(caretaker_id=current_user.id).order_by(Booking.start_time.desc()).all()
    
    # Group bookings by status
    pending_bookings = [b for b in bookings if b.status == 'pending']
    confirmed_bookings = [b for b in bookings if b.status == 'confirmed']
    completed_bookings = [b for b in bookings if b.status == 'completed']
    cancelled_bookings = [b for b in bookings if b.status == 'cancelled']

    return render_template('my_bookings.html',
                         pending_bookings=pending_bookings,
                         confirmed_bookings=confirmed_bookings,
                         completed_bookings=completed_bookings,
                         cancelled_bookings=cancelled_bookings,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


# Context processor to provide notifications to all templates
@app.context_processor
def inject_notifications():
    if not current_user.is_authenticated:
        return {}
    
    all_notifications = []
    
    # For pet owners: show service requests that have been accepted by caretakers
    if current_user.user_type == 'pet_owner':
        accepted_requests = PetServiceRequest.query.filter_by(
            pet_owner_id=current_user.id,
            status='accepted'
        ).order_by(PetServiceRequest.created_at.desc()).limit(10).all()
        
        for req in accepted_requests:
            caretaker = User.query.get(req.caretaker_id)
            pet = Pet.query.get(req.pet_id)
            all_notifications.append({
                'id': req.id,
                'type': 'service_accepted',
                'message': f'{caretaker.name} accepted your service request for {pet.name}',
                'time': req.created_at,
                'link': url_for('dashboard')
            })
    
    # For caretakers: show open service requests from pet owners
    elif current_user.user_type == 'caretaker':
        open_requests = PetServiceRequest.query.filter_by(
            status='open'
        ).order_by(PetServiceRequest.created_at.desc()).limit(10).all()
        
        for req in open_requests:
            owner = User.query.get(req.pet_owner_id)
            pet = Pet.query.get(req.pet_id)
            all_notifications.append({
                'id': req.id,
                'type': 'service_request',
                'message': f'{owner.name} needs services for {pet.name}',
                'time': req.created_at,
                'link': url_for('list_service_requests')
            })
    
    # Sort notifications by time (newest first)
    all_notifications.sort(key=lambda x: x['time'], reverse=True)
    
    # Get read notification IDs for the current user
    read_notifications = ReadNotification.query.filter_by(user_id=current_user.id).all()
    read_items = [(rn.notification_type, rn.notification_id) for rn in read_notifications]
    
    # Filter out read notifications
    unread_notifications = []
    for notification in all_notifications:
        if (notification['type'], notification['id']) not in read_items:
            unread_notifications.append(notification)
    
    # Include both unread and all notifications
    return {
        'notifications': unread_notifications,
        'all_notifications': all_notifications,
        'read_notification_pairs': read_items
    }

# Create all database tables
with app.app_context():
    db.create_all()

# Load models when app starts
load_models()

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        # Get form data
        name = request.form.get('name')
        email = request.form.get('email')
        location = request.form.get('location')
        
        # Update user information
        current_user.name = name
        current_user.email = email
        current_user.location = location
        
        # Commit changes to database
        db.session.commit()
        
        flash('Profile updated successfully!', 'success')
        return redirect(url_for('caretaker_profile', user_id=current_user.id))
    
    return redirect(url_for('dashboard'))

# Route to delete a service request
@app.route('/delete-service-request/<int:request_id>', methods=['POST'])
@login_required
def delete_service_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is the owner of the service request
    if service_request.pet_owner_id != current_user.id:
        flash('You are not authorized to delete this service request.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Delete the service request
    db.session.delete(service_request)
    db.session.commit()
    
    flash('Service request has been deleted successfully.', 'success')
    return redirect(url_for('dashboard'))


# Route to reject a service request
@app.route('/reject-service-request/<int:request_id>', methods=['POST'])
@login_required
def reject_service_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is a caretaker
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can reject service requests', 'danger')
        return redirect(url_for('dashboard'))
    
    # Update the service request status to cancelled
    service_request.status = 'cancelled'
    db.session.commit()
    
    flash('Service request rejected', 'info')
    return redirect(url_for('dashboard'))


# Route for pet owners to accept caretaker requests
@app.route('/owner-accept-request/<int:request_id>', methods=['POST'])
@login_required
def owner_accept_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is the pet owner
    if current_user.id != service_request.pet_owner_id:
        flash('Only the pet owner can accept this request', 'danger')
        return redirect(url_for('dashboard'))
    
    # Check if the request is in pending status
    if service_request.status != 'pending':
        flash('This request cannot be accepted in its current state', 'warning')
        return redirect(url_for('view_service_request', request_id=request_id))
    
    # Update the service request status to accepted
    service_request.status = 'accepted'
    db.session.commit()
    
    # Get pet and caretaker information
    pet = Pet.query.get(service_request.pet_id)
    caretaker = User.query.get(service_request.caretaker_id)
    
    # Create notification for caretaker
    notification_message = f"Your request to take care of {pet.name} has been accepted by the owner."
    create_notification(service_request.caretaker_id, notification_message)
    
    flash('You have accepted the caretaker request', 'success')
    return redirect(url_for('view_service_request', request_id=request_id))


# Route for pet owners to decline caretaker requests
@app.route('/owner-decline-request/<int:request_id>', methods=['POST'])
@login_required
def owner_decline_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is the pet owner
    if current_user.id != service_request.pet_owner_id:
        flash('Only the pet owner can decline this request', 'danger')
        return redirect(url_for('dashboard'))
    
    # Check if the request is in pending status
    if service_request.status != 'pending':
        flash('This request cannot be declined in its current state', 'warning')
        return redirect(url_for('view_service_request', request_id=request_id))
    
    # Update the service request status to declined
    service_request.status = 'declined'
    db.session.commit()
    
    # Get pet and caretaker information
    pet = Pet.query.get(service_request.pet_id)
    caretaker = User.query.get(service_request.caretaker_id)
    
    # Create notification for caretaker
    notification_message = f"Your request to take care of {pet.name} has been declined by the owner."
    create_notification(service_request.caretaker_id, notification_message)
    
    flash('You have declined the caretaker request', 'info')
    return redirect(url_for('view_service_request', request_id=request_id))


# Route to mark notifications as read
@app.route('/mark-notification-read', methods=['POST'])
@login_required
def mark_notification_read():
    try:
        data = request.get_json()
        notification_id = data.get('id')
        
        if not notification_id:
            return jsonify({'success': False, 'message': 'Missing notification ID'}), 400
        
        # Find the notification
        notification = Notification.query.filter_by(
            id=notification_id,
            user_id=current_user.id
        ).first()
        
        if notification:
            # Mark notification as read
            notification.is_read = True
            db.session.commit()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'No notification found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Route to display available pets for caretakers to book
@app.route('/available_pets_for_booking')
@login_required
def available_pets_for_booking():
    # Check if the user is a caretaker
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can view available pets', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get all pets
    pets = Pet.query.all()
    
    # Get all open service requests
    open_requests = PetServiceRequest.query.filter_by(status='open').all()
    
    return render_template('available_pets.html', pets=pets, open_requests=open_requests)


# Route for caretakers to book pets available for care
@app.route('/book_pet_for_care/<int:pet_id>', methods=['GET', 'POST'])
@login_required
def book_pet_for_care(pet_id):
    # Check if the user is a caretaker
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can book pets for care', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the pet
    pet = Pet.query.get_or_404(pet_id)
    
    # Get the pet owner
    pet_owner = User.query.get(pet.owner_id)
    
    if request.method == 'POST':
        # Get form data
        start_date = request.form.get('start_time')
        end_date = request.form.get('end_time')
        services_needed = request.form.getlist('services_needed')
        special_instructions = request.form.get('special_instructions')
        
        # Create a new service request
        service_request = PetServiceRequest(
            pet_id=pet_id,
            caretaker_id=current_user.id,
            pet_owner_id=pet.owner_id,
            start_time=datetime.strptime(start_date, '%Y-%m-%dT%H:%M') if start_date else datetime.now(),
            end_time=datetime.strptime(end_date, '%Y-%m-%dT%H:%M') if end_date else (datetime.now() + timedelta(days=1)),
            services_needed=json.dumps(services_needed),
            description=special_instructions,
            status='pending'
        )
        
        # Add to database
        db.session.add(service_request)
        db.session.commit()
        
        # Create notification for pet owner
        caretaker_name = current_user.name
        
        # Format dates for display in notification
        formatted_start = service_request.start_time.strftime('%b %d, %Y')
        formatted_end = service_request.end_time.strftime('%b %d, %Y')
        
        notification_message = f"{caretaker_name} has requested to take care of your pet {pet.name} from {formatted_start} to {formatted_end}."
        create_notification(pet.owner_id, notification_message)
        
        flash('Service request submitted successfully!', 'success')
        return redirect(url_for('dashboard'))
    
    # Get notification data
    notifications_data = get_notifications()
    
    return render_template('book_pet.html', 
                          pet=pet, 
                          pet_owner=pet_owner,
                          unread_notifications=notifications_data['unread_notifications'],
                          all_notifications=notifications_data['all_notifications'])


# Route to view service request details
@app.route('/service_request_details/<int:request_id>')
@login_required
def service_request_details(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is authorized to view this service request
    # Allow caretakers to view open requests, and allow pet owners and assigned caretakers to view their own requests
    is_authorized = (
        # Caretakers can view any open request
        (current_user.user_type == 'caretaker' and service_request.status == 'open') or
        # Pet owners can view their own requests
        (current_user.id == service_request.pet_owner_id) or
        # Assigned caretakers can view their assigned requests
        (service_request.caretaker_id and current_user.id == service_request.caretaker_id)
    )
    
    if not is_authorized:
        flash('You do not have permission to view this service request', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get notification data
    notifications_data = get_notifications()
    
    return render_template('view_service_request.html', 
                         request=service_request,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


# Route to view all open requests from pet owners
@app.route('/open_service_requests')
@login_required
def view_open_service_requests():
    if current_user.user_type != 'caretaker':
        flash('Only caretakers can view service requests', 'warning')
        return redirect(url_for('dashboard'))

    # Get all open service requests
    open_requests = PetServiceRequest.query.filter_by(status='open').all()
    
    # Get notification data
    notifications_data = get_notifications()
    
    return render_template('open_service_requests.html', 
                         open_requests=open_requests,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


# Route to decline a service request
@app.route('/decline_service_request/<int:request_id>', methods=['POST'])
@login_required
def decline_service_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the request is open (not already accepted or declined)
    if service_request.status != 'open':
        flash('This request has already been processed', 'warning')
        return redirect(url_for('list_service_requests'))
    
    # Get the pet owner for notification
    pet_owner_id = service_request.pet_owner_id
    pet_name = service_request.pet.name
    
    # Mark the service request as declined
    service_request.status = 'declined'
    db.session.commit()
    
    # Create notification for pet owner
    notification_message = f"{current_user.name} has declined the service request for your pet {pet_name}."
    create_notification(pet_owner_id, notification_message)
    
    flash('Service request has been declined', 'info')
    return redirect(url_for('list_service_requests'))


# Route to view booking details
@app.route('/view_booking/<int:booking_id>')
@login_required
def view_booking(booking_id):
    # Get the booking
    booking = Booking.query.get_or_404(booking_id)
    
    # Check if the current user is authorized to view this booking
    if current_user.id != booking.caretaker_id and current_user.id != booking.pet_owner_id:
        flash('You do not have permission to view this booking', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get notification data
    notifications_data = get_notifications()
    
    return render_template('view_booking.html', 
                         booking=booking,
                         unread_notifications=notifications_data['unread_notifications'],
                         all_notifications=notifications_data['all_notifications'])


# Routes for accept_booking and decline_booking have been removed to fix the duplicate route definition issue


# Route to cancel a booking
@app.route('/cancel_booking/<int:booking_id>', methods=['POST'])
@login_required
def cancel_booking(booking_id):
    # Get the booking
    booking = Booking.query.get_or_404(booking_id)
    
    # Check if the current user is the caretaker of this booking
    if current_user.id != booking.caretaker_id:
        flash('You do not have permission to cancel this booking', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the pet owner for notification
    pet_owner_id = booking.pet_owner_id
    pet_name = booking.pet.name
    
    # Delete the booking
    db.session.delete(booking)
    db.session.commit()
    
    # Create notification for pet owner
    notification_message = f"{current_user.name} has canceled the booking for your pet {pet_name}."
    create_notification(pet_owner_id, notification_message)
    
    flash('Booking has been canceled successfully', 'success')
    return redirect(url_for('dashboard'))


# Route to mark a service request as complete
@app.route('/mark_service_request_complete/<int:request_id>', methods=['POST'])
@login_required
def mark_service_request_complete(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is the caretaker of this request
    if current_user.id != service_request.caretaker_id:
        flash('You do not have permission to mark this service request as complete', 'danger')
        return redirect(url_for('list_service_requests'))
    
    # Get the pet owner for notification
    pet_owner_id = service_request.pet_owner_id
    pet_name = service_request.pet.name
    
    # Mark the service request as completed
    service_request.status = 'completed'
    db.session.commit()
    
    # Create notification for pet owner
    notification_message = f"{current_user.name} has marked the service request for your pet {pet_name} as completed."
    create_notification(pet_owner_id, notification_message)
    
    flash('Service request has been marked as completed successfully', 'success')
    return redirect(url_for('list_service_requests'))


# Route to delete a service request
@app.route('/cancel_service_request/<int:request_id>', methods=['POST'])
@login_required
def cancel_service_request(request_id):
    # Get the service request
    service_request = PetServiceRequest.query.get_or_404(request_id)
    
    # Check if the current user is the caretaker of this request
    if current_user.id != service_request.caretaker_id:
        flash('You do not have permission to delete this service request', 'danger')
        return redirect(url_for('list_service_requests'))
    
    # Get the pet owner for notification
    pet_owner_id = service_request.pet_owner_id
    pet_name = service_request.pet.name
    
    # Delete the service request
    db.session.delete(service_request)
    db.session.commit()
    
    # Create notification for pet owner
    notification_message = f"{current_user.name} has cancelled the service request for your pet {pet_name}."
    create_notification(pet_owner_id, notification_message)
    
    flash('Service request has been deleted successfully', 'success')
    return redirect(url_for('list_service_requests'))


# Function to create database tables if they don't exist
def initialize_database():
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")

if __name__ == '__main__':
    # Initialize database
    initialize_database()
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)