# 🎁 Gift Intent Analyzer

An **AI-powered gift recommendation system** that uses semantic vector embeddings to suggest the perfect gift bundle based on relationship dynamics, demographics, and personal vibes.

## Features

✨ **AI-Powered Recommendations** - Uses Sentence Transformers for semantic understanding  
🎯 **Demographic-Aware** - Considers age group, gender, and profession  
📦 **Smart Bundling** - "Rule of Three" approach (Anchor, Complement, Filler)  
💰 **Budget-Friendly** - Works within your spending constraints  
🎨 **Modern UI** - Beautiful gradient design with smooth animations  
⚡ **Fast & Responsive** - Real-time recommendations powered by vector search  

## Tech Stack

- **Backend**: Flask, Python, Sentence Transformers, Pandas
- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **ML/AI**: Semantic embeddings, vector similarity search
- **Architecture**: REST API with JSON payloads

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/gift-analyzer.git
cd gift-analyzer
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # macOS/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Open in browser**
Navigate to `http://127.0.0.1:5000`

## How It Works

### 1. User Input
Users provide:
- **Relationship**: Partner, Boss, Friend, Parent, etc.
- **Occasion**: Birthday, Anniversary, Graduation, etc.
- **Age Group**: Teen, Young Adult, Adult, Senior
- **Gender**: Male, Female, Neutral
- **Profession/Hobby**: Software Engineer, Gardener, Artist, etc.
- **Vibe**: Personality description (stressed workaholic, romantic, etc.)
- **Budget**: Maximum spending amount

### 2. Prompt Stuffing
The system constructs a rich natural language query:
```
"A {vibe} gift for a {age_group} {gender} who is a {profession}. 
Relationship: {relation}. Occasion: {occasion}."
```

### 3. Vector Encoding
The query is encoded into semantic embeddings using Sentence Transformers (`all-MiniLM-L6-v2`)

### 4. Semantic Search
The system searches through 30+ gift items with rich tags covering:
- Professions (engineer, chef, student, etc.)
- Demographics (teen, senior, male, female, etc.)
- Context (professional, casual, luxury, wellness, etc.)

### 5. Smart Bundling (Rule of Three)
Returns a 3-item bundle:
- **Anchor** (⚓): Best match, up to 70% of budget
- **Complement** (✨): Complementary item for remaining budget
- **Filler** (🍬): Small gift under $20 for final touches

## Project Structure

```
gift-analyzer/
├── app.py                    # Flask backend
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── README.md                # This file
├── gift_analyzer_model/     # Pre-trained model
│   ├── config.json
│   ├── model.safetensors
│   ├── modules.json
│   └── tokenizer.json
└── templates/
    ├── index.html           # Main HTML
    ├── styles.css           # 900+ lines of modern CSS
    ├── main.js              # Application logic
    └── utils.js             # Helper functions
```

## API Endpoints

### POST `/predict`
Generates gift recommendations based on user input.

**Request:**
```json
{
  "relation": "Partner",
  "occasion": "Birthday",
  "age_group": "Adult",
  "gender": "Female",
  "profession": "Software Engineer",
  "vibe": "Stressed workaholic who loves coffee",
  "budget": 100
}
```

**Response:**
```json
{
  "status": "success",
  "bundle": [
    {
      "id": 1,
      "name": "Noise Cancelling Headphones",
      "price": 150,
      "tags": "..."
    },
    ...
  ],
  "total_cost": 85,
  "intent_analysis": "Analysis: Optimized for 'Stressed workaholic who loves coffee' vibe with 92% confidence."
}
```

## CSS Features (40+ Design Elements)

- Gradient backgrounds and text effects
- Glassmorphism (backdrop blur)
- Smooth animations and transitions
- Hover effects and interactive states
- Responsive grid layouts
- Mobile-first design
- Custom CSS variables for theming
- Shimmer effects on cards
- Pop-in animations for badges
- Staggered animations for items

## JavaScript Features

- Real-time form validation
- localStorage preference persistence
- Keyboard shortcuts (Ctrl+Enter to submit)
- Debounce and throttle utilities
- Error handling and recovery
- Analytics event tracking
- Smooth scroll behavior
- Responsive behavior detection

## Customization

### Add More Gifts
Edit `app.py` and add items to the `raw_data` list:
```python
{"id": 31, "name": "Item Name", "price": 50, 
 "tags": "keyword1 keyword2 keyword3 ... keyword15"}
```

### Change Theme Colors
Edit CSS variables in `styles.css`:
```css
:root {
    --primary: #6366f1;
    --secondary: #ec4899;
    --accent: #f59e0b;
    /* ... */
}
```

## Performance Optimization

- Caching of embeddings with pickle
- Lazy loading of assets
- Efficient vector search (top-k similarity)
- Minimal DOM manipulation
- CSS animations use GPU acceleration

## Future Enhancements

- [ ] User accounts and saved preferences
- [ ] Rating system for recommendations
- [ ] More gift categories (100+ items)
- [ ] Multi-language support
- [ ] Mobile app
- [ ] Integration with shopping APIs
- [ ] Analytics dashboard

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Author

Created with ❤️ for thoughtful gifting

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with AI-powered vector embeddings and modern web technologies** 🚀
