class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    user_type = db.Column(db.String(20), nullable=False)  # 'pet_owner' or 'caretaker'
    phone = db.Column(db.String(15), nullable=False)
    alternate_phone = db.Column(db.String(15))
    address = db.Column(db.Text, nullable=False)
    city = db.Column(db.String(50), nullable=False)
    state = db.Column(db.String(50), nullable=False)
    pincode = db.Column(db.String(6), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    pets = db.relationship('Pet', backref='owner', lazy=True)
    profile = db.relationship('CaretakerProfile', backref='user', uselist=False, lazy=True)
    service_requests = db.relationship('PetServiceRequest', backref='pet_owner', lazy=True)
    accepted_requests = db.relationship('PetServiceRequest', backref='caretaker', lazy=True, foreign_keys='PetServiceRequest.caretaker_id')
    
    def __repr__(self):
        return f'<User {self.email}>' 


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