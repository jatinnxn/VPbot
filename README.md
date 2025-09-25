# VP Bot – Visa & Passport Assistant 🤖✈️

## 📌 Overview
VP Bot is a **chatbot assistant** that helps users with visa and passport queries.  
It provides quick links for visa applications and passport services, using data from simple CSV files.  

The chatbot runs **locally in the browser** using Python + FastAPI.  

---

## 📦 Folder Contents
vp_bot/
├── vp_bot.py                # Main script (backend + frontend merged)
├── requirements.txt         # Python dependencies
├── country_aliases.csv      # Maps country aliases (e.g., "UK" → "united kingdom")
├── data/
│ └── country_links.csv      # Maps countries → visa links
└── logo.png # Chatbot logo (optional)

yaml
Copy code

---

## ⚙️ Requirements
- Python **3.9+**
- `pip` (Python package manager)

---

## 🚀 Setup & Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
Run the bot

bash
Copy code
python vp_bot.py
Access the chatbot

The bot will automatically open in your browser at:
👉 http://localhost:8000

🛠️ Customization
Change logo → Replace logo.png with your own image.

Update visa links → Edit data/country_links.csv and add/update countries + links.

Update aliases → Edit country_aliases.csv to map alternative country names.

📞 Support
If you face any issues, please contact the project maintainer.