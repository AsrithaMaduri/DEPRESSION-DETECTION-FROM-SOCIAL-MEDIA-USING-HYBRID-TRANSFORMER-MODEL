from app import create_app
from app.database import db, User

app = create_app()

with app.app_context():
    user = User.query.filter_by(email="ashrithamaduri@gmail.com").first()

    if user:
        user.is_admin = True
        db.session.commit()
        print("User is now admin")
    else:
        print("User not found")