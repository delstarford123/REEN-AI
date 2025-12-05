import os
import sys
import requests
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# --- IMPORTS FOR CLOUDINARY ---
import cloudinary
import cloudinary.uploader
import cloudinary.api
from cloudinary.utils import cloudinary_url

# --- 1. SETUP PATHS & SECRETS ---
load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(current_dir, 'reen', 'script')
sys.path.append(script_path)

try:
    from predict import predict_molecule
except ImportError:
    print("⚠️ Warning: predict.py not found.")
    def predict_molecule(s): return {"error": "Prediction script not found"}

app = Flask(__name__)

# --- 2. CONFIGURATION ---
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/analyze_chem', methods=['POST'])
def analyze_chem():
    """
    Intelligent Biochemistry Route:
    1. Accepts Name (e.g., 'Aspirin') OR SMILES.
    2. Converts Name -> SMILES using PubChem.
    3. If Chemical: Runs ML Prediction + PubChem Image.
    4. If Biological (e.g., 'Red Blood Cell'): Runs Gemini Description + AI Image.
    """
    data = request.get_json()
    user_input = data.get('smiles', '').strip()
    
    smiles = None
    canonical_name = user_input
    is_biological = False

    # --- STEP 1: RESOLVE NAME TO STRUCTURE ---
    # Try PubChem API to turn "Aspirin" into SMILES
    try:
        pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{user_input}/property/CanonicalSMILES/JSON"
        req = requests.get(pubchem_url)
        
        if req.status_code == 200:
            # It's a known chemical!
            res_json = req.json()
            smiles = res_json['PropertyTable']['Properties'][0]['CanonicalSMILES']
            canonical_name = user_input.title()
        else:
            # PubChem didn't find it. 
            # Check if user actually typed a SMILES string (has special chars like =, #, @)
            if any(char in user_input for char in ['=', '#', '@', '[', ']']):
                 smiles = user_input
            else:
                 # It's likely a biological term (e.g., "Red Blood Cells")
                 is_biological = True
    except:
        is_biological = True

    # --- STEP 2: CHEMICAL ANALYSIS (If we found SMILES) ---
    if smiles and not is_biological:
        # Run your Custom ML Model
        result = predict_molecule(smiles)
        
        # Get real chemical structure image
        mol_image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{requests.utils.quote(smiles)}/PNG"
        result['image_url'] = mol_image_url
        result['input_name'] = canonical_name
        
        return jsonify(result)

    # --- STEP 3: BIOLOGICAL FALLBACK (The "Study Buddy" Mode) ---
    else:
        # If it's "Red Blood Cells", our chemical model can't classify it.
        # So we use Gemini to define it and Pollinations to show it.
        
        desc = "Biological Entity"
        if GEMINI_API_KEY:
            try:
                # Ask Gemini for a 1-sentence definition
                gem_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
                gem_payload = { "contents": [{ "parts": [{ "text": f"Define {user_input} in biochemistry in one short sentence." }] }] }
                gem_res = requests.post(gem_url, json=gem_payload).json()
                desc = gem_res['candidates'][0]['content']['parts'][0]['text']
            except:
                desc = "Complex Biological Structure"

        # Generate an educational image using Pollinations
        ai_image_url = f"https://image.pollinations.ai/prompt/microscopic {user_input} scientific diagram white background?width=500&height=500&nologo=true"

        return jsonify({
            "classification": "Biological / Complex",
            "confidence": 100,
            "smiles": desc, # We use the SMILES field to show the description in the UI
            "image_url": ai_image_url
        })

@app.route('/generate_image', methods=['POST'])
def generate_image():
    """Generates Art using Pollinations (Reliable Fallback) -> Saves to Cloudinary"""
    data = request.get_json()
    prompt = data.get('prompt')
    
    print(f"🎨 Generating Art for: {prompt}")

    # 1. Generate via Pollinations
    ai_image_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=1024&height=1024&nologo=true"

    try:
        # 2. Download and Upload to Cloudinary
        response = requests.get(ai_image_url)
        if response.status_code == 200:
            upload_result = cloudinary.uploader.upload(response.content, folder="reen_gallery")
            return jsonify({
                'success': True,
                'cloudinary_url': upload_result['secure_url'],
                'public_id': upload_result['public_id']
            })
        return jsonify({'error': "AI Generation timed out."}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/transform_image', methods=['POST'])
def transform_image():
    """Applies Cloudinary filters"""
    data = request.get_json()
    public_id = data.get('public_id')
    effect = data.get('effect') 

    transformation = []
    
    if effect == 'biochem_art':
        transformation = [{'effect': "art:hokusai"}, {'border': "5px_solid_white"}]
    elif effect == 'sketch':
         transformation = [{'effect': "cartoonify"}, {'effect': "outline:100"}]
    elif effect == 'glam':
         transformation = [{'effect': "vignette:50"}, {'effect': "improve"}]
    elif effect == 'love':
         transformation = [{'effect': "vignette:30"}, {'overlay': "text:roboto_40_bold:For Reen", 'gravity': "south", 'y': 20, 'color': "pink"}]

    new_url, _ = cloudinary_url(public_id, transformation=transformation)
    return jsonify({'new_url': new_url})

if __name__ == '__main__':
    print("🌸 Reen AI is starting on http://127.0.0.1:5000")
    app.run(debug=True)