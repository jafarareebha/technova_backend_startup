import os
import sys

# Add app dir to path
sys.path.append(r'c:\Users\zaina\Downloads\technova-main\technova-main\technova-backend')
os.chdir(r'c:\Users\zaina\Downloads\technova-main\technova-main\technova-backend')

from app import app, db, Analysis, predict_startup
import json

with app.app_context():
    # Grab the latest analysis
    a = Analysis.query.order_by(Analysis.id.desc()).first()
    if not a:
        print("No analysis found")
        sys.exit()
    
    inputs = json.loads(a.inputs)
    print("BASE INPUTS:", inputs)
    
    score1, cat1, fac1 = predict_startup(inputs)
    print("\nBASE PREDICTION:", score1, fac1)
    
    # Modify inputs
    inputs['initial_funding'] = float(inputs.get('initial_funding', 50000)) * 2.0
    inputs['team_size'] = int(inputs.get('team_size', 5)) + 15
    
    print("\nMODIFIED INPUTS:", inputs['initial_funding'], inputs['team_size'])
    score2, cat2, fac2 = predict_startup(inputs)
    print("MODIFIED PREDICTION:", score2, fac2)
