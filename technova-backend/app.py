from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import json
import os
import google.genai as genai
import PyPDF2
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()
print(f"DEBUG: API Key loaded: {os.getenv('GEMINI_API_KEY') is not None}")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

def fallback_recommendations(score, factors, category):
    """Fallback strategy if Gemini generation fails"""
    recommendations = []
    
    if factors.get('Market', 50) < 50:
        recommendations.append({"icon": "📊", "title": "Market Pivot Required", "description": "Consider pivoting to a larger or faster-growing market segment. Your market score suggests limited opportunities."})
    else:
        recommendations.append({"icon": "📊", "title": "Aggressive Market Capture", "description": "Your market selection shows promise. Focus on capturing market share through targeted marketing."})
    
    if factors.get('Team', 50) < 50:
        recommendations.append({"icon": "👥", "title": "Strategic Hiring Needed", "description": "Strengthen your team with experienced advisors or key hires in business development and technology."})
    else:
        recommendations.append({"icon": "👥", "title": "Scale Your Solid Team", "description": "Your team composition is solid. Consider adding complementary skills for scaling."})
    
    if factors.get('Product', 50) < 50:
        recommendations.append({"icon": "💡", "title": "Iterate Product-Market Fit", "description": "Focus on product-market fit through customer discovery and iterative development."})
    else:
        recommendations.append({"icon": "💡", "title": "Accelerate Development", "description": "Your product concept is strong. Accelerate development and gather user feedback."})
    
    if factors.get('Financials', 50) < 50:
        recommendations.append({"icon": "💰", "title": "Extend Financial Runway", "description": "Explore alternative funding sources or adjust burn rate to extend runway."})
    else:
        recommendations.append({"icon": "💰", "title": "Pursue Growth Funding", "description": "Your financial foundation looks solid. Consider growth-stage funding options."})
    
    if factors.get('Competition', 50) < 50:
        recommendations.append({"icon": "⚔️", "title": "Establish Differentiation", "description": "Develop stronger differentiation from competitors. Identify unique value propositions."})
    else:
        recommendations.append({"icon": "⚔️", "title": "Defend Market Position", "description": "You have competitive advantages. Defend them with IP and strategic partnerships."})
    
    return {
        "tags": [
            {"text": "Growth priority", "color": "blue"},
            {"text": "Market expansion", "color": "blue"},
            {"text": "Partnership strategy", "color": "blue"}
        ],
        "recommendations": recommendations[:5]
    }

def generate_dynamic_strategy(form_data, factors, score, category):
    """Dynamic strategy generation using Gemini based on idea description and analysis."""
    print("DEBUG: generate_dynamic_strategy called with updated code")
    description = (form_data.get('idea_description') or '')[:12000].strip()
    if not description:
        description = "No specific description provided. Provide strategic advice based exclusively on the provided industry, metrics, and numerical scores."
        
    summary = json.dumps({
        'overall_score': score,
        'category': category,
        'factors': factors,
        'startup_name': form_data.get('startup_name', 'Unknown'),
        'industry_sector': form_data.get('industry_sector', 'Unknown'),
        'market_type': form_data.get('market_type', 'Unknown')
    }, indent=0)

    prompt = f"""You are an expert startup advisor. Based ONLY on the startup description and the model summary below, provide a practical, tailored strategy and suggestions.
    
Startup description:
{description}

Model summary:
{summary}

Return ONLY valid JSON with this exact structure (no markdown wrapper, no extra text, just the valid JSON object):
{{
    "tags": [
        {{"text": "e.g., Growth risk", "color": "red"}},
        {{"text": "e.g., Market expansion", "color": "blue"}}
    ],
    "recommendations": [
        {{
            "icon": "📊",
            "title": "A highly specific, custom sub-heading based on their unique market (e.g., 'Hyper-Niche SaaS Targeting')",
            "description": "Specific paragraph of advice..."
        }},
        {{
            "icon": "👥",
            "title": "...",
            "description": "..."
        }}
    ]
}}

Rules:
- Give exactly 3 tags relevant to their strategy. Use color 'red' for risk/warning and 'blue' for positive/opportunity/neutral.
- Give exactly 5 recommendations. Ensure the 'icon' is an appropriate emoji (e.g. 📊, 👥, 💡, 💰, ⚔️).
- The 'title' attribute MUST be heavily customized and highly specific to their exact idea, not just 'Market Strategy'.
- The advice MUST be highly specific to their exact startup idea description and scores provided, NOT generic boilerplate. Even if no description is provided, use the industry, market type, and specific scores to give distinct analysis.
"""
    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        text = (response.text or '').strip()
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "tags" in data and "recommendations" in data:
                print("✅ Successfully generated dynamic strategy!")
                return data
            else:
                print("❌ Generated strategy missing required keys.")
        else:
            print("❌ Strategy generation returned malformed JSON text.")
    except Exception as e:
        print(f"❌ Strategy generation failed: {e}")
        error_dict = fallback_recommendations(score, factors, category)
        error_dict['recommendations'].insert(0, {
            "icon": "⚠️",
            "title": "API Error Detected",
            "description": f"Gemini API generation failed with error: {str(e)}. Please restart your server if you just updated your .env file."
        })
        return error_dict
        
    return fallback_recommendations(score, factors, category)



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
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        
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

def fallback_risks_resources(form_data, factors, score, category):
    """Rule-based risks & resource areas when Gemini is unavailable."""
    funding = float(form_data.get('initial_funding', 0) or 0)
    team = int(form_data.get('team_size', 1) or 1)
    comp_n = int(form_data.get('competitors', 0) or 0)
    risks = []
    if factors.get('Competition', 50) < 55:
        risks.append({
            'title': 'Intense Market Competition',
            'description': 'You have many competitors. You must clearly highlight what makes your product unique, find better ways to reach customers, or reward user loyalty to stand out.',
            'severity': 'high' if factors.get('Competition', 50) < 40 else 'medium',
        })
    if factors.get('Financials', 50) < 55 or funding < 75000:
        risks.append({
            'title': 'Limited Funding and Cash Runway',
            'description': 'Your current budget might run out quickly relative to your team size. You may not have enough time to perfect the product before you need to secure more money or find paying users.',
            'severity': 'medium',
        })
    if int(form_data.get('acquisition_difficulty', 5) or 5) >= 7:
        risks.append({
            'title': 'High Cost to Find Customers',
            'description': 'It will be difficult and expensive to convince people to try your product. You should carefully plan your marketing channels and partner with others to reduce costs.',
            'severity': 'high',
        })
    if int(form_data.get('market_demand', 5) or 5) <= 4:
        risks.append({
            'title': 'Uncertain Customer Demand',
            'description': 'It is not completely clear if enough users truly want this solution. You should definitely talk to potential users and run small tests before building the full product.',
            'severity': 'medium',
        })
    if category in ('Weak', 'Poor'):
        risks.append({
            'title': 'Overall Business Feasibility',
            'description': 'Our model gave this idea a lower score overall. We highly recommend rethinking your core target audience or your pricing strategy before spending significant money.',
            'severity': 'high',
        })
    if not risks:
        risks.append({
            'title': 'Execution and Scaling',
            'description': 'There are no major flashing warnings right now. Focus entirely on disciplined execution, tracking your metrics, and hitting your initial milestones.',
            'severity': 'low',
        })

    runway_months = max(1, int(funding / max(1, team * 8000)))
    resource_areas = [
        {
            'title': 'Capital & Budget Tracker',
            'items': [
                f"Estimated Timeline: You have roughly ~{runway_months} months of operations before running out of money (this varies based on real salaries).",
                'Action: Plot out exactly what you absolutely must pay for right away versus what can wait, keeping 3–6 months of emergency cash.',
            ],
        },
        {
            'title': 'People & Team Skills',
            'items': [
                f"Team Size Planned: {team}. Make sure your next hire directly helps you hit your immediate milestone (like a salesperson if you need revenue, or a developer for the product).",
                'Action: Find part-time advisors or freelancers for specific complex tasks you lack (like legal compliance or deep financial planning).',
            ],
        },
        {
            'title': 'Operations & Software Tools',
            'items': [
                'Action: Keep the first version of the product as simple as possible. Decide exactly which affordable software tools you will use for basic hosting and tracking users.',
                'Action: Write down how you will handle basic daily tasks early on, so the business runs smoothly as you slowly add new users.',
            ],
        },
    ]
    if comp_n > 60:
        resource_areas[0]['items'].append('Budget for competitive intelligence and positioning work (not only ads).')
    return {'risks': risks, 'resource_areas': resource_areas, 'source': 'fallback'}


def generate_risks_resources(form_data, factors, score, category):
    """
    Input-grounded risk register and resource checklist via Gemini, with fallback.
    """
    description = (form_data.get('idea_description') or '')[:12000].strip()
    if not description:
        description = "No specific description provided. Assess potential risks and required resources based exclusively on the provided industry, metrics, team size, and initial funding."

    summary = json.dumps({
        'overall_score': score,
        'category': category,
        'factors': factors,
        'initial_funding': form_data.get('initial_funding'),
        'team_size': form_data.get('team_size'),
        'competitors': form_data.get('competitors'),
        'industry_sector': form_data.get('industry_sector'),
        'market_type': form_data.get('market_type'),
        'country_region': form_data.get('country_region'),
        'market_demand': form_data.get('market_demand'),
        'acquisition_difficulty': form_data.get('acquisition_difficulty'),
        'barriers': form_data.get('barriers'),
    }, indent=0)

    prompt = f"""You are advising a new entrepreneur. Based ONLY on the startup description and the model summary below, provide a highly readable, jargon-free risk register and a practical resource checklist.
Use proper business concepts, but explain them in simple, easy-to-understand language. Avoid complex buzzwords. Ensure any beginner founder could immediately understand what to do next.

Startup description:
{description}

Model summary (use as context, do not invent external market data):
{summary}

Return ONLY valid JSON with this exact structure (no markdown):
{{
  "risks": [
    {{"title": "Simple short title", "description": "1-3 easy-to-understand sentences explaining what could go wrong and how to avoid it.", "severity": "high" or "medium" or "low"}}
  ],
  "resource_areas": [
    {{"title": "e.g. Budget & Cash Flow", "items": ["Clear, actionable checklist item", "..."]}},
    {{"title": "e.g. Team Hiring Needs", "items": ["...", "..."]}},
    {{"title": "e.g. Tools & Setup", "items": ["...", "..."]}}
  ]
}}

Rules:
- 4-6 distinct risks directly related to the idea. Explain the risk clearly without overcomplicating it.
- 3-5 resource_areas with 2-4 items each. Items must be immediately actionable (e.g., "Set aside 3 months of emergency cash" rather than "Optimize runway capitalization").
- severity must be exactly lowercase: high, medium, or low.
"""

    try:
        response = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
        text = (response.text or '').strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            risks = data.get('risks') or []
            areas = data.get('resource_areas') or []
            if isinstance(risks, list) and isinstance(areas, list) and risks and areas:
                out = {'risks': [], 'resource_areas': [], 'source': 'gemini'}
                for r in risks[:8]:
                    if not isinstance(r, dict):
                        continue
                    sev = (r.get('severity') or 'medium').lower()
                    if sev not in ('high', 'medium', 'low'):
                        sev = 'medium'
                    out['risks'].append({
                        'title': str(r.get('title', 'Risk'))[:200],
                        'description': str(r.get('description', ''))[:1200],
                        'severity': sev,
                    })
                for a in areas[:6]:
                    if not isinstance(a, dict):
                        continue
                    items = a.get('items') or []
                    if not isinstance(items, list):
                        items = []
                    out['resource_areas'].append({
                        'title': str(a.get('title', 'Resources'))[:120],
                        'items': [str(x)[:500] for x in items[:8]],
                    })
                if out['risks'] and out['resource_areas']:
                    return out
    except Exception as e:
        print(f"❌ Risks/resources generation failed: {e}")
        error_dict = fallback_risks_resources(form_data, factors, score, category)
        error_dict['risks'].insert(0, {
            'title': 'API Error Detected',
            'description': f"Gemini API generation failed with error: {str(e)}. Please restart your server if you just updated your .env file.",
            'severity': 'high'
        })
        return error_dict

    return fallback_risks_resources(form_data, factors, score, category)

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
    
    # Generate and store dynamic strategy and risk recommendations
    form_data['strategy_suggestions'] = generate_dynamic_strategy(form_data, factors, score, category)
    form_data['risks_resources'] = generate_risks_resources(form_data, factors, score, category)
    
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
    
    # Check if loaded strategy is the fallback version
    strategy = inputs.get('strategy_suggestions')
    is_strategy_fallback = False
    if strategy and isinstance(strategy, dict) and strategy.get('tags') and len(strategy['tags']) > 0:
        if strategy['tags'][0].get('text') == 'Growth priority':
            is_strategy_fallback = True

    if not strategy or is_strategy_fallback:
        inputs['strategy_suggestions'] = generate_dynamic_strategy(inputs, factors, analysis.score, analysis.category)
        needs_commit = True
        
    # Check if loaded risks is the fallback version
    risks = inputs.get('risks_resources')
    is_risks_fallback = False
    if risks and isinstance(risks, dict) and risks.get('source') == 'fallback':
        is_risks_fallback = True
        
    if not risks or is_risks_fallback:
        inputs['risks_resources'] = generate_risks_resources(inputs, factors, analysis.score, analysis.category)
        needs_commit = True
        
    if needs_commit:
        analysis.inputs = json.dumps(inputs)
        db.session.commit()
    
    analysis_data = {
        'id': analysis.id,
        'startup_name': analysis.startup_name,
        'score': analysis.score,
        'category': analysis.category,
        'factors': factors,
        'inputs': inputs,
        'risks_resources': inputs.get('risks_resources') or {},
        'strategy_suggestions': inputs.get('strategy_suggestions') or fallback_recommendations(analysis.score, factors, analysis.category),
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
        
        # Determine the initial baseline factors to ensure smooth relative changes instead of capping
        orig_factors = json.loads(analysis.factors)
        
        # Dynamic ML Physics Modifier: Decision Trees usually flatline on minor perturbations.
        # We apply this ON TOP of the newly predicted ML score so it responds fluidly.
        modifier = (funding_pct / 100.0) * 12.0
        modifier -= (mkt_pct / 100.0) * 3.0
        modifier += team_delta * 1.5
        modifier -= (comp_val - 5) * 2.0
        
        sim_score = min(99.0, max(1.0, score + modifier))
        
        factors['Financials'] = min(99.0, max(1.0, orig_factors.get('Financials', 50) + (funding_pct / 10.0)))
        factors['Team'] = min(99.0, max(1.0, orig_factors.get('Team', 50) + team_delta * 4.0))
        factors['Market'] = min(99.0, max(1.0, orig_factors.get('Market', 50) + (mkt_pct / 15.0) - (comp_val - 5) * 2.5))
        factors['Competition'] = min(99.0, max(1.0, orig_factors.get('Competition', 50) - (comp_val - 5) * 5.0))
        factors['Product'] = min(99.0, max(1.0, orig_factors.get('Product', 50) + (funding_pct / 20.0)))
        
        return jsonify({
            'score': sim_score,
            'category': score_to_category(sim_score),
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
    
    # Generate strategy using the dynamic function
    strategy_suggestions = inputs.get('strategy_suggestions') or generate_dynamic_strategy(inputs, factors, analysis.score, analysis.category)
    
    # Store in session
    session['current_analysis'] = {
        'id': analysis.id,
        'startup_name': analysis.startup_name,
        'score': analysis.score,
        'category': analysis.category,
        'factors': factors,
        'strategy_suggestions': strategy_suggestions,
        'inputs': inputs
    }
    
    return redirect(url_for('results', analysis_id=analysis.id))

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
    app.run(host="0.0.0.0",port=10000, debug=True)