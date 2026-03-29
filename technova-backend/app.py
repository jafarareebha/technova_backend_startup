from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import json
import os
import google.generativeai as genai
import PyPDF2
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Warning: Gemini API Key not configured properly: {e}")

app = Flask(__name__)
model = pickle.load(open("startup_model.pkl","rb"))
app.config['SECRET_KEY'] = 'your-secret-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ===== DATABASE MODELS =====
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # In production, hash this!
    analyses = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    startup_name = db.Column(db.String(200), nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    score = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(20), nullable=False)
    
    # Store all inputs and factors as JSON
    inputs = db.Column(db.Text, nullable=False)  # JSON string
    factors = db.Column(db.Text, nullable=False)  # JSON string
    
    # Store individual fields for easy display
    industry = db.Column(db.String(100))
    market_type = db.Column(db.String(100))
    initial_funding = db.Column(db.Float)
    team_size = db.Column(db.Integer)
    country = db.Column(db.String(100))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== LOAD ML MODEL =====
def load_model():
    """Load the trained model"""
    try:
        with open('startup_model.pkl', 'rb') as f:
            saved = pickle.load(f)
        print("✅ Model loaded successfully!")
        return saved
    except FileNotFoundError:
        print("❌ Model file not found. Please ensure startup_model.pkl is in the same directory.")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load model at startup
model_data = load_model()

# ===== CATEGORY FUNCTION =====
def score_to_category(score):
    """Convert score to category based on your model"""
    if score >= 75:
        return "Strong"
    elif score >= 55:
        return "Moderate"
    elif score >= 35:
        return "Weak"
    else:
        return "Poor"

# ===== PREDICTION FUNCTION =====
def predict_startup(form_data):
    """
    Takes form data, returns score and category
    Uses your exact model structure
    """
    if model_data is None:
        # Fallback if model not loaded
        print("⚠️ Using fallback prediction (model not loaded)")
        score = np.random.randint(40, 95)
        category = score_to_category(score)
        
        # Generate factor scores (simulated)
        factors = {
            'Market': min(100, max(0, score + np.random.randint(-10, 10))),
            'Team': min(100, max(0, score + np.random.randint(-10, 10))),
            'Product': min(100, max(0, score + np.random.randint(-10, 10))),
            'Financials': min(100, max(0, score + np.random.randint(-10, 10))),
            'Competition': min(100, max(0, score + np.random.randint(-10, 10)))
        }
        return score, category, factors
    
    try:
        # Extract model components
        model = model_data['model']
        encoders = model_data['encoders']
        feature_cols = model_data['feature_cols']
        
        # Build input row for prediction
        row = {}
        
        # Handle categorical columns with encoding
        cat_cols = ['industry_sector', 'market_type', 'country_region']
        for col in cat_cols:
            val = form_data.get(col, 'Unknown')
            le = encoders.get(col)
            if le and val in le.classes_:
                row[col + '_enc'] = int(le.transform([val])[0])
            else:
                row[col + '_enc'] = 0  # fallback for unseen categories
        
        # Add all other features
        feature_mapping = {
            'founding_team_size': 'team_size',
            'initial_funding_usd': 'initial_funding',
            'num_funding_rounds': 'funding_rounds',
            'num_competitors_proxy': 'competitors',
            'year_founded': 'year_founded',
            'market_demand': 'market_demand',
            'pain_point_severity': 'pain_point',
            'idea_novelty': 'novelty',
            'scalability': 'scalability',
            'barriers_to_entry': 'barriers',
            'revenue_model_strength': 'revenue_strength',
            'acquisition_difficulty': 'acquisition_difficulty',
            'willingness_to_pay': 'willingness_to_pay'
        }
        
        for model_feat, form_key in feature_mapping.items():
            row[model_feat] = float(form_data.get(form_key, 0))
        
        # Create DataFrame with all feature columns in correct order
        input_df = pd.DataFrame([row])
        
        # Ensure all feature columns exist (fill missing with 0)
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Make prediction
        prediction = model.predict(input_df[feature_cols])[0]
        score = float(prediction)
        score = round(min(max(score, 0), 100), 1)
        category = score_to_category(score)
        
        # Calculate factor scores based on feature contributions
        # This is a simplified approach - you can make this more sophisticated
        factors = {
            'Market': min(100, max(0, score + (float(form_data.get('market_demand', 5)) - 5) * 3)),
            'Team': min(100, max(0, score + (float(form_data.get('team_size', 5)) - 5) * 2)),
            'Product': min(100, max(0, score + (float(form_data.get('novelty', 5)) - 5) * 4)),
            'Financials': min(100, max(0, score + (float(form_data.get('initial_funding', 50000)) / 50000 - 1) * 10)),
            'Competition': min(100, max(0, score - float(form_data.get('competitors', 50)) / 10))
        }
        
        print(f"✅ Prediction successful: Score={score}, Category={category}")
        return score, category, factors
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        # Fallback
        score = 65
        category = "Moderate"
        factors = {
            'Market': 70, 'Team': 65, 'Product': 60, 
            'Financials': 55, 'Competition': 75
        }
        return score, category, factors

def generate_recommendations(score, factors, category, inputs):
    """Generate strategy recommendations based on scores"""
    recommendations = []
    
    if factors.get('Market', 50) < 50:
        recommendations.append("📊 **Market Strategy**: Consider pivoting to a larger or faster-growing market segment. Your market score suggests limited opportunities.")
    else:
        recommendations.append("📊 **Market Strategy**: Your market selection shows promise. Focus on capturing market share through targeted marketing.")
    
    if factors.get('Team', 50) < 50:
        recommendations.append("👥 **Team Building**: Strengthen your team with experienced advisors or key hires in business development and technology.")
    else:
        recommendations.append("👥 **Team Building**: Your team composition is solid. Consider adding complementary skills for scaling.")
    
    if factors.get('Product', 50) < 50:
        recommendations.append("💡 **Product Development**: Focus on product-market fit through customer discovery and iterative development.")
    else:
        recommendations.append("💡 **Product Development**: Your product concept is strong. Accelerate development and gather user feedback.")
    
    if factors.get('Financials', 50) < 50:
        recommendations.append("💰 **Financial Strategy**: Explore alternative funding sources or adjust burn rate to extend runway.")
    else:
        recommendations.append("💰 **Financial Strategy**: Your financial foundation looks solid. Consider growth-stage funding options.")
    
    if factors.get('Competition', 50) < 50:
        recommendations.append("⚔️ **Competitive Strategy**: Develop stronger differentiation from competitors. Identify unique value propositions.")
    else:
        recommendations.append("⚔️ **Competitive Strategy**: You have competitive advantages. Defend them with IP and strategic partnerships.")
    
    # Category-specific advice
    if category == "Strong":
        recommendations.append("🚀 **Growth Stage**: You're well-positioned for rapid growth. Focus on scaling operations and building brand.")
    elif category == "Moderate":
        recommendations.append("📈 **Development Stage**: Good foundation. Address key weaknesses before major scaling.")
    elif category == "Weak":
        recommendations.append("⚠️ **Improvement Needed**: Significant gaps to address. Consider refining your business model.")
    else:
        recommendations.append("🔴 **Critical Attention Required**: Major challenges exist. Reassess core assumptions.")
    
    return recommendations[:5]  # Return top 5 recommendations


# ===== NLP GEMINI EXTRACTOR =====
def extract_startup_parameters(description):
    """Uses Gemini to extract 8 subjective parameters from startup description"""
    if not description or len(description) < 15:
        return {}
    
    prompt = f"""
    Analyze the following startup business description and predict scores on a scale of 1 to 10 for the following 8 metrics.
    Be objective, realistic, and critical based on the information provided.
    Return ONLY a valid JSON object with the exact keys below.
    
    Keys to evaluate (1-10):
    - market_demand (Demand from target audience)
    - pain_point (Severity of the problem solved)
    - novelty (Uniqueness of the idea)
    - scalability (Potential to grow without proportional cost)
    - barriers (Barriers to entry for new competitors, high is better for the startup)
    - revenue_strength (Viability of the revenue model)
    - acquisition_difficulty (Difficulty to acquire customers, 10=very high difficulty)
    - willingness_to_pay (Customer's willingness to pay for the solution)
    
    Startup Description:
    {description}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        
        # Parse JSON from response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            print("✅ Gemini successfully extracted parameters:", data)
            return data
    except Exception as e:
        print(f"❌ Gemini evaluation failed: {e}")
    
    return {}

# ===== ROUTES =====
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        # Simple password check (in production, use hashed passwords!)
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('register'))
        
        # Check if user exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        # Create new user
        new_user = User(username=username, password=password)  # Hash password in production!
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect("https://capiche-alpha.vercel.app")

@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's analyses, newest first
    analyses = Analysis.query.filter_by(user_id=current_user.id)\
                             .order_by(Analysis.date.desc())\
                             .all()
    return render_template('dashboard.html', analyses=analyses)


@app.route('/analyze', methods=['GET'])
@login_required
def new_analysis():
        return render_template('form.html')
    
@app.route('/analyze/start', methods=['POST'])
@login_required
def analyze():
    # Process PDF if provided
    pdf_text = ""
    if 'idea_pdf' in request.files:
        file = request.files['idea_pdf']
        if file.filename != '':
            try:
                reader = PyPDF2.PdfReader(file)
                pdf_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            except Exception as e:
                print(f"PDF extraction error: {e}")
                
    idea_description = request.form.get('idea_description', '')
    combined_text = idea_description + "\n\n" + pdf_text
    
    # Extract AI parameters
    ai_params = extract_startup_parameters(combined_text)
    
    # Get form data using AI params as fallback overrides
    form_data = {
        'startup_name': request.form.get('startup_name'),
        'industry_sector': request.form.get('industry_sector'),
        'market_type': request.form.get('market_type'),
        'country_region': request.form.get('country_region'),
        'initial_funding': float(request.form.get('initial_funding', 50000)),
        'team_size': int(request.form.get('team_size', 5)),
        'funding_rounds': int(request.form.get('funding_rounds', 1)),
        'competitors': int(request.form.get('competitors', 50)),
        'year_founded': int(request.form.get('year_founded', 2024)),
        'market_demand': int(ai_params.get('market_demand', 5)),
        'pain_point': int(ai_params.get('pain_point', 5)),
        'novelty': int(ai_params.get('novelty', 5)),
        'scalability': int(ai_params.get('scalability', 5)),
        'barriers': int(ai_params.get('barriers', 5)),
        'revenue_strength': int(ai_params.get('revenue_strength', 5)),
        'acquisition_difficulty': int(ai_params.get('acquisition_difficulty', 5)),
        'willingness_to_pay': int(ai_params.get('willingness_to_pay', 5)),
        'idea_description': combined_text,
        'ai_predictions_used': True if ai_params else False
    }
    
    # Get prediction
    score, category, factors = predict_startup(form_data)
    
    # Generate recommendations
    recommendations = generate_recommendations(score, factors, category, form_data)
    
    # Save to database
    import json

    new_analysis_obj = Analysis(
    user_id=current_user.id,
    startup_name=form_data['startup_name'],
    score=score,
    category=category,
    inputs=json.dumps(form_data),
    factors=json.dumps(factors),
    industry=form_data['industry_sector'],
    market_type=form_data['market_type'],
    initial_funding=form_data['initial_funding'],
    team_size=form_data['team_size'],
    country=form_data['country_region']
    )

    db.session.add(new_analysis_obj)
    db.session.commit()
    
    return redirect(url_for('results', analysis_id=new_analysis_obj.id))

@app.route('/results/<int:analysis_id>')
@login_required
def results(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Ensure user owns this analysis
    if analysis.user_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    # Load analysis data
    inputs = json.loads(analysis.inputs)
    factors = json.loads(analysis.factors)
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis.score, factors, analysis.category, inputs)
    
    analysis_data = {
        'id': analysis.id,
        'startup_name': analysis.startup_name,
        'score': analysis.score,
        'category': analysis.category,
        'factors': factors,
        'recommendations': recommendations,
        'inputs': inputs
    }
    
    return render_template('results.html', analysis=analysis_data)

@app.route('/simulate', methods=['POST'])
@login_required
def simulate():
    """API endpoint for real-time simulation using ML model"""
    try:
        data = request.json
        analysis_id = data.get('analysis_id')
        
        analysis = Analysis.query.get(analysis_id)
        if not analysis or analysis.user_id != current_user.id:
            return jsonify({'error': 'Unauthorized or Not Found'}), 404
            
        # Get original form inputs
        base_inputs = json.loads(analysis.inputs)
        
        # Apply deltas from frontend
        funding_pct = float(data.get('funding_delta_pct', 0))
        base_inputs['initial_funding'] = float(base_inputs.get('initial_funding', 0)) * (1 + (funding_pct / 100.0))
        
        team_delta = int(data.get('team_delta', 0))
        base_inputs['team_size'] = max(1, int(base_inputs.get('team_size', 1)) + team_delta)
        
        comp_val = int(data.get('competition_val', 5))
        base_inputs['competitors'] = comp_val * 10  # roughly scale 1-10 to 10-100
        
        mkt_pct = float(data.get('marketing_delta_pct', 0))
        acq_diff = float(base_inputs.get('acquisition_difficulty', 5))
        new_acq = max(1, acq_diff - (mkt_pct / 50.0))
        base_inputs['acquisition_difficulty'] = round(new_acq)
        
        # Run true ML Prediction
        score, category, factors = predict_startup(base_inputs)
        
        return jsonify({
            'score': score,
            'category': category,
            'factors': factors
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/load_analysis/<int:analysis_id>')
@login_required
def load_analysis(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Ensure user owns this analysis
    if analysis.user_id != current_user.id:
        flash('Access denied')
        return redirect(url_for('dashboard'))
    
    # Load analysis data
    inputs = json.loads(analysis.inputs)
    factors = json.loads(analysis.factors)
    
    # Generate recommendations
    recommendations = generate_recommendations(analysis.score, factors, analysis.category, inputs)
    
    # Store in session
    session['current_analysis'] = {
        'id': analysis.id,
        'startup_name': analysis.startup_name,
        'score': analysis.score,
        'category': analysis.category,
        'factors': factors,
        'recommendations': recommendations,
        'inputs': inputs
    }
    
    return redirect(url_for('results'))

# ===== CREATE DATABASE =====
with app.app_context():
    db.create_all()
    
    # Create a test user (remove in production)
    if not User.query.filter_by(username='demo').first():
        test_user = User(username='demo', password='demo123')
        db.session.add(test_user)
        db.session.commit()
        print("✅ Created demo user: demo / demo123")

# ===== RUN THE APP =====
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Startup Analyzer Web App")
    print("="*50)
    if model_data:
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Running in fallback mode (model not loaded)")
    print("📝 Demo login: demo / demo123")
    print("="*50 + "\n")
    app.run(host="0.0.0.0",port=10000)