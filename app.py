from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import pickle

# Create Flask app with static folder pointing to templates
app = Flask(__name__, static_folder='templates', static_url_path='')

# --- GLOBAL VARIABLES ---
model = None
df = None
embeddings = None

# --- 1. INITIALIZATION LOGIC ---
def initialize_system():
    """
    Loads the model and data.
    If a saved pickle exists, load it. If not, generate from scratch.
    """
    global model, df, embeddings
    
    print("⏳ Initializing Gift Analyzer System...")
    
    try:
        # Check if we have saved data
        if os.path.exists('inventory_data.pkl'):
            print("🔹 Loading from disk...")
            with open('inventory_data.pkl', 'rb') as f:
                data = pickle.load(f)
                df = data['dataframe']
                embeddings = data['embeddings']
            # Load model (assuming it's cached or saved)
            # using 'all-MiniLM-L6-v2' triggers a download if not found locally
            model = SentenceTransformer('all-MiniLM-L6-v2') 
        else:
            print("🔹 First run detected. Generating database & embeddings...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # --- EXPANDED INVENTORY WITH DETAILED TAGS (Professions & Demographics) ---
            raw_data = [
                 {"id": 1, "name": "Noise Cancelling Headphones", "price": 150, "tags": "tech engineer software developer focus travel work quiet music premium professional concentration student adult"},
                 {"id": 2, "name": "Weighted Blanket", "price": 70, "tags": "home sleep anxiety comfort cozy warm heavy therapy relaxation senior adult mental-health wellness bedtime"},
                 {"id": 3, "name": "Leather Desk Mat", "price": 40, "tags": "office work desk professional aesthetic architect designer student executive clean organized workspace luxury"},
                 {"id": 4, "name": "Gourmet Truffle Box", "price": 40, "tags": "food sweet luxury chocolate romantic delicious confectionery adult female gourmet indulgence celebration"},
                 {"id": 5, "name": "Funny Socks", "price": 12, "tags": "fun clothes cheap joke gag-gift teen young-adult gen-z humor casual workplace personality"},
                 {"id": 6, "name": "Keychain Multi-tool", "price": 10, "tags": "utility small cheap pocket gadget engineer outdoors camping hiking survival practical professional"},
                 {"id": 7, "name": "Scented Soy Candle", "price": 22, "tags": "home decor smell romantic relax warm cozy adult female interior-design aromatherapy wellness evening"},
                 {"id": 8, "name": "Moleskine Notebook", "price": 25, "tags": "office writing notes journal classic quality student architect author writer professional organized thinker"},
                 {"id": 9, "name": "Stress Ball", "price": 5, "tags": "office toy small cheap anxiety relief focus fidget teen student professional desk work casual"},
                 {"id": 10, "name": "Cocktail Shaker Set", "price": 35, "tags": "drink alcohol party bar mixology adult entertainer home-bar social gathering celebration professional"},
                 {"id": 11, "name": "Gourmet Coffee Bag", "price": 15, "tags": "drink food small morning boost caffeine professional barista coffee-lover adult morning-routine work"},
                 {"id": 12, "name": "Luxury Perfume Sampler", "price": 60, "tags": "personal scent luxury beauty romantic female adult young-adult fragrance collection sophisticated elegant"},
                 {"id": 13, "name": "Bath Bomb Set", "price": 18, "tags": "personal spa relax bath cheap fun self-care female adult wellness aromatherapy home-spa luxury"},
                 {"id": 14, "name": "Wireless Charging Pad", "price": 30, "tags": "tech gadget smartphone charge convenience modern young-adult professional student minimalist office desk"},
                 {"id": 15, "name": "Herb Garden Kit", "price": 45, "tags": "gardening cooking home indoor farmer chef adult senior hobby plant green-thumb organic kitchen"},
                 {"id": 16, "name": "Blue Light Glasses", "price": 55, "tags": "tech health eye-care professional student programmer developer computer-worker wellness adult modern"},
                 {"id": 17, "name": "Portable Bluetooth Speaker", "price": 65, "tags": "tech audio music entertainment young-adult teen outdoor party social gathering portable travel"},
                 {"id": 18, "name": "Silk Sleep Mask", "price": 20, "tags": "personal sleep luxury travel female adult wellness beauty skincare comfortable premium"},
                 {"id": 19, "name": "Bamboo Cutting Board Set", "price": 35, "tags": "kitchen cooking chef home adult professional gourmet food-lover sustainable bamboo wood"},
                 {"id": 20, "name": "Yoga Mat with Carrying Strap", "price": 50, "tags": "fitness wellness yoga exercise female adult young-adult health mindfulness portable gym"},
                 {"id": 21, "name": "Premium Dark Chocolate Box", "price": 28, "tags": "food gourmet luxury chocolate adult romantic celebration special-occasion sweet indulgence premium"},
                 {"id": 22, "name": "Desk Organizer Set", "price": 25, "tags": "office organization workspace professional student neat tidy adult architect designer workplace"},
                 {"id": 23, "name": "Handmade Soap Gift Set", "price": 32, "tags": "personal bath skincare luxury female adult wellness organic natural self-care premium"},
                 {"id": 24, "name": "Polaroid Instant Camera", "price": 85, "tags": "photography hobby artist creative young-adult teen memories nostalgia social sharing fun"},
                 {"id": 25, "name": "Electric Wine Opener", "price": 35, "tags": "drink wine adult entertainment home-bar gathering celebration modern convenience gadget"},
                 {"id": 26, "name": "Weighted Eye Mask", "price": 25, "tags": "sleep wellness relax comfort adult female mindfulness meditation stress-relief premium"},
                 {"id": 27, "name": "Specialty Tea Collection", "price": 30, "tags": "drink beverage connoisseur adult relaxation morning ritual wellness gourmet premium variety"},
                 {"id": 28, "name": "Minimalist Watch", "price": 95, "tags": "fashion accessory professional adult male young-adult workplace elegant simple luxury modern"},
                 {"id": 29, "name": "Leather Bookmark Set", "price": 15, "tags": "book reading student adult writer author intellectual classic quality vintage premium"},
                 {"id": 30, "name": "Diffuser Humidifier", "price": 45, "tags": "home wellness aromatherapy air-quality adult female family health bedroom living-room modern"}
            ]
            
            df = pd.DataFrame(raw_data)
            # Create Embeddings
            df['search_text'] = df['name'] + " " + df['tags']
            embeddings = model.encode(df['search_text'].tolist(), convert_to_tensor=True)
            print("✅ Data generated successfully.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR during initialization: {e}")

# Run initialization immediately
initialize_system()

# --- 2. ROUTES ---

@app.route('/')
def home():
    """Serves the main webpage."""
    return render_template('index.html')

def convert_to_json_serializable(obj):
    """Convert numpy and pandas types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

@app.route('/predict', methods=['POST'])
def predict():
    """API Endpoint: Receives JSON user intent, returns JSON gift bundle with prompt stuffing."""
    try:
        # Get data from frontend
        data = request.json
        relation = data.get('relation', '')
        occasion = data.get('occasion', '')
        age_group = data.get('age_group', '')
        gender = data.get('gender', '')
        profession = data.get('profession', '')
        vibe = data.get('vibe', '')
        budget = float(data.get('budget', 0))

        if not all([relation, occasion, age_group, gender, profession, vibe, budget]):
            return jsonify({"error": "Missing fields. Please fill all inputs."}), 400

        # 1. Vector Search with Prompt Stuffing
        # Construct a detailed natural language query that includes all demographic information
        user_query = f"A {vibe} gift for a {age_group} {gender} who is a {profession}. Relationship: {relation}. Occasion: {occasion}."
        query_vec = model.encode(user_query, convert_to_tensor=True)
        
        # Get top 15 matches to have a pool to choose from
        hits = util.semantic_search(query_vec, embeddings, top_k=15)
        
        # Process hits
        matched_indices = [hit['corpus_id'] for hit in hits[0]]
        scores = [hit['score'] for hit in hits[0]]
        
        candidates = df.iloc[matched_indices].copy()
        candidates['score'] = scores
        
        # 2. Filter by Budget
        candidates = candidates[candidates['price'] <= budget]
        
        if candidates.empty:
            return jsonify({"error": "Budget too low for this specific request."}), 400

        # 3. Clustering Logic (Rule of Three)
        bundle = []
        current_cost = 0
        candidates = candidates.sort_values(by='score', ascending=False)
        anchor_pool = candidates[candidates['price'] <= (budget * 0.7)]
        if anchor_pool.empty: anchor_pool = candidates # Fallback
        
        anchor = anchor_pool.iloc[0]
        bundle.append(convert_to_json_serializable(anchor.to_dict()))
        current_cost += anchor['price']
        
        # B. Complement (Next best, fits remaining budget)
        candidates = candidates[candidates['id'] != anchor['id']] # Remove used
        rem_budget = budget - current_cost
        
        comp_pool = candidates[candidates['price'] <= rem_budget]
        if not comp_pool.empty:
            comp = comp_pool.iloc[0]
            bundle.append(convert_to_json_serializable(comp.to_dict()))
            current_cost += comp['price']
            candidates = candidates[candidates['id'] != comp['id']]
            
        # C. Filler (Cheap item <$20)
        rem_budget = budget - current_cost
        fill_pool = candidates[(candidates['price'] <= rem_budget) & (candidates['price'] <= 20)]
        if not fill_pool.empty:
            filler = fill_pool.iloc[0]
            bundle.append(convert_to_json_serializable(filler.to_dict()))
            current_cost += filler['price']

        # Return success JSON
        return jsonify({
            "status": "success",
            "bundle": bundle,
            "total_cost": int(current_cost),
            "intent_analysis": f"Analysis: Optimized for '{vibe}' vibe with {int(anchor['score']*100)}% confidence."
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "Internal Server Error. Check console logs."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)