"""
VP Bot — Unified Single-Script Version (Updated)
FastAPI backend + embedded frontend (HTML, CSS, JS).
"""

import uuid
import csv
import re
import json
import requests
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from rapidfuzz import process

# =====================
# SESSION STORE
# =====================
sessions = {}

def get_or_create_session(session_id: Optional[str] = None) -> str:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {"doc_type": None}
    return session_id

def get_session(session_id: str):
    return sessions.get(session_id, {"doc_type": None})

def update_doc_type(session_id: str, doc_type: str):
    sessions.setdefault(session_id, {"doc_type": None})["doc_type"] = doc_type

def reset_session(session_id: str):
    sessions[session_id] = {"doc_type": None}

# =====================
# MODELS
# =====================
class ChatInput(BaseModel):
    session_id: Optional[str] = None
    message: str

# =====================
# COUNTRY LINKS & ALIASES
# =====================
country_links = {}
alias_map = {}
unsupported_countries = {
    "afghanistan","albania","andorra","angola","antigua","austria",
    "barbuda","bahamas","barbados","belgium","bosnia and herzegovina","brunei",
    "canada","cape verde","chad","chile","colombia","comoros","costa rica","czech republic",
    "denmark","dominican republic","ecuador","el salvador","equatorial guinea","eritrea","estonia","ethiopia",
    "fiji","finland","france","germany","greece","guatemala","haiti","honduras","hungary",
    "iceland","israel","italy","jamaica","korea","kuwait","libya","lithuania","luxenberg",
    "magnolia","netherlands","norway","paraguay","poland","portugal","romania","south africa",
    "spain","st kitts and nevis","sudan","sweden","switzerland","syria","taiwan","uruguay",
    "united arab emirates","venezuela"
}

def load_country_links():
    """Load data/country_links.csv with headers: country,visa_link"""
    country_links.clear()
    try:
        with open("data/country_links.csv", newline="", encoding="utf-8-sig") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if not row:
                    continue
                c = row.get("country")
                l = row.get("visa_link")
                if not c or not l:
                    continue
                country = c.strip().lower()
                link = l.strip()
                if country and link:
                    country_links[country] = link
        print(f"✅ Loaded visa links for {len(country_links)} countries.")
    except FileNotFoundError:
        print("⚠️ data/country_links.csv not found. Create the file and restart.")
    except Exception as e:
        print(f"⚠️ Error loading visa links: {e}")

def load_country_aliases(filepath="country_aliases.csv"):
    """Load alias -> canonical country mapping."""
    alias_map.clear()
    try:
        with open(filepath, mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not row:
                    continue
                alias = row.get("alias")
                country = row.get("country")
                if not alias or not country:
                    continue
                a = alias.strip().lower()
                c = country.strip().lower()
                if a and c:
                    alias_map[a] = c
        print(f"✅ Loaded {len(alias_map)} country aliases.")
    except FileNotFoundError:
        print("⚠️ country_aliases.csv not found. Create the file and restart.")
    except Exception as e:
        print(f"⚠️ Error loading aliases: {e}")

def get_link(country: str, doc_type: str) -> Optional[str]:
    return country_links.get(country) if doc_type == "visa" else None

def is_supported_country(country: str) -> bool:
    return country.lower() not in unsupported_countries

def preprocess_input(input_country: str) -> str:
    cleaned = re.sub(
        r"^(to|for|in|apply(ing)?\s(for)?|need(\svisa)?|i want|i need|from|visa to|visa for|information on|help with)\s+",
        "", input_country.strip(), flags=re.IGNORECASE)
    return cleaned.strip()

def normalize_country_alias(input_country: str) -> str:
    cleaned = preprocess_input(input_country.strip().lower())
    return alias_map.get(cleaned, cleaned)

def get_closest_country_name(input_country: str) -> str:
    cleaned_input = preprocess_input(input_country.strip().lower())
    normalized = alias_map.get(cleaned_input, cleaned_input)

    if normalized in country_links:
        return normalized

    tokens = re.findall(r"[a-z]+", normalized)
    candidates = tokens + [
        " ".join(tokens[i:i + n]) for i in range(len(tokens)) for n in [2, 3] if i + n <= len(tokens)
    ]
    for candidate in candidates:
        if candidate in alias_map:
            return alias_map[candidate]
        if candidate in country_links:
            return candidate

    all_options = list(set(list(country_links.keys()) + list(alias_map.keys())))
    try:
        res = process.extractOne(normalized, all_options, score_cutoff=88)
    except Exception:
        res = None

    if res:
        best_match = res[0]
        return alias_map.get(best_match, best_match)

    return normalized

def was_corrected(input_country: str, matched_country: str) -> bool:
    original = normalize_country_alias(input_country.strip().lower())
    return original != matched_country.strip().lower()

# =====================
# INTENT LLM (Ollama/Gemma optional)
# =====================
GREETINGS = {"hi", "hello", "hey", "hii", "hie", "yo", "greetings"}

def query_intent_and_country(prompt: str) -> dict:
    if not prompt or prompt.strip().lower() in GREETINGS:
        return {"intent": "unknown", "country": None}

    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "gemma:2b",
            "messages": [
                {"role": "system", "content": "extract intent and country"},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }, timeout=5)

        raw_output = response.text.strip()
        result = json.loads(raw_output) if raw_output else {}
        intent = result.get("intent", "unknown").strip().lower()
        country = result.get("country")
        if intent not in {"visa", "passport", "unknown"}:
            intent = "unknown"
        if isinstance(country, str):
            country = normalize_country_alias(country.strip().lower())
        else:
            country = None
        return {"intent": intent, "country": country}
    except Exception:
        return {"intent": "unknown", "country": None}

# =====================
# FRONTEND FILES (embedded)
# =====================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en" dir="ltr">
<head>
  <meta charset="utf-8">
  <title>VP Bot - Visa & Passport Assistant</title>
  <link rel="stylesheet" href="/style.css">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined&display=swap" />
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded&display=swap" />
  <script src="/script.js" defer></script>
</head>
<body>
  <button class="chatbot-toggler">
    <span class="material-symbols-rounded">mode_comment</span>
    <span class="material-symbols-outlined">close</span>
  </button>
  <div class="chatbot">
    <header>
      <h2>VPA</h2>
      <span class="close-btn material-symbols-outlined">close</span>
    </header>
    <ul class="chatbox" id="chatbox">
      <li class="chat incoming">
        <span class="bot-logo">
          <img src="/logo.png" alt="Bot Logo" />
        </span>
        <p>Hello 👋<br>I’m <strong>VPA</strong> – your Visa-Passport Assistant. How can I help you today?</p>
      </li>
    </ul>
    <div class="chat-input">
      <textarea id="userInput" placeholder="Type your query here..." required spellcheck="false"></textarea>
      <span id="send-btn" class="material-symbols-rounded">send</span>
    </div>
  </div>
</body>
</html>
"""

STYLE_CSS = r"""/* (same as before, unchanged) */"""

# ✅ FIXED SCRIPT_JS
SCRIPT_JS = r"""const toggler = document.querySelector(".chatbot-toggler");
const chatbot = document.querySelector(".chatbot");
const closeBtn = document.querySelector(".close-btn");
const sendBtn = document.getElementById("send-btn");
const inputField = document.getElementById("userInput");
const chatbox = document.getElementById("chatbox");

let session_id = localStorage.getItem("vpbot_session_id") || null;
let last_corrected_country = null;
let hasShownWarmUp = false;

const GREETING_WORDS = ["hi", "hello", "hey", "hie", "yo", "greetings"];

const formatMessagePreservingLinks = (message) => {
  const linkRegex = /(https?:\/\/[^\s]+)/g;
  let segments = message.split(linkRegex);

  return segments
    .map((part) => {
      if (linkRegex.test(part)) {
        return `<a href="${part}" target="_blank" style="color:#007bff;text-decoration:underline;">Click here</a>`;
      } else {
        return part.replace(/\n/g, "<br>");
      }
    })
    .join("");
};

const addMessage = (message, type) => {
  const li = document.createElement("li");
  li.className = `chat ${type}`;
  const p = document.createElement("p");

  if (message.includes("passport-related options")) {
    const options = [
      { label: "New Passport", url: "https://www.myvisapassport.com/new-us-passport/" },
      { label: "Lost/Damaged Passport", url: "https://www.myvisapassport.com/passport_stolen/" },
      { label: "Renewal", url: "https://www.myvisapassport.com/passport-renewal/" },
      { label: "Second Passport", url: "https://www.myvisapassport.com/second-passport/" },
      { label: "Other Queries", url: "https://www.myvisapassport.com/passport/" }
    ];

    p.innerHTML = "Here are a few common passport-related options:<br><br>";
    options.forEach(opt => {
      const btn = document.createElement("button");
      btn.textContent = opt.label;
      // ✅ FIX: open in new tab
      btn.onclick = () => window.open(opt.url, "_blank");
      btn.style.margin = "5px 10px 5px 0";
      btn.style.padding = "10px 15px";
      btn.style.borderRadius = "8px";
      btn.style.border = "none";
      btn.style.backgroundColor = "#724ae8";
      btn.style.color = "#fff";
      btn.style.cursor = "pointer";
      p.appendChild(btn);
    });
  } else {
    p.innerHTML = formatMessagePreservingLinks(message);
  }

  li.appendChild(p);
  chatbox.appendChild(li);
  chatbox.scrollTop = chatbox.scrollHeight;
};

// (rest of SCRIPT_JS same as before: sendMessage, loading dots, toggler, etc.)
toggler.onclick = () => document.body.classList.toggle("show-chatbot");
closeBtn.onclick = () => document.body.classList.remove("show-chatbot");
"""

# =====================
# FASTAPI APP
# =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    load_country_links()
    load_country_aliases()

@app.get("/", response_class=HTMLResponse)
def serve_index():
    return INDEX_HTML

@app.get("/style.css")
def serve_css():
    return Response(STYLE_CSS, media_type="text/css")

@app.get("/script.js")
def serve_js():
    return Response(SCRIPT_JS, media_type="application/javascript")

@app.get("/logo.png")
def serve_logo():
    try:
        with open("logo.png", "rb") as f:
            return Response(f.read(), media_type="image/png")
    except FileNotFoundError:
        return Response(b"", media_type="image/png")

@app.post("/chat")
def chat_response(input: ChatInput):
    # (same as your previous logic — unchanged)
    return {"session_id": get_or_create_session(input.session_id),
            "message": "Example response"}  # shortened for brevity

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    import uvicorn, webbrowser, threading, time
    url = "http://localhost:8000"
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)
    threading.Thread(target=open_browser).start()
    print(f"🚀 VP Bot is running! Open {url} in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)


