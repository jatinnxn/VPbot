"""
VP Bot — Unified Single-Script Version
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

    # direct match
    if normalized in country_links:
        return normalized

    # token-based heuristics
    tokens = re.findall(r"[a-z]+", normalized)
    candidates = tokens + [
        " ".join(tokens[i:i + n]) for i in range(len(tokens)) for n in [2, 3] if i + n <= len(tokens)
    ]
    for candidate in candidates:
        if candidate in alias_map:
            return alias_map[candidate]
        if candidate in country_links:
            return candidate

    # fuzzy match as last resort
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
    """
    Returns: {"intent": "visa"/"passport"/"unknown", "country": <lowercase country or None>}
    If Ollama/local LLM is unavailable or parsing fails, returns unknown.
    """
    if not prompt or prompt.strip().lower() in GREETINGS:
        return {"intent": "unknown", "country": None}

    system_prompt = (
        "You are a backend API assistant. Your job is to extract exactly two lowercase fields from a user's message:\n"
        "- 'intent': one of 'visa', 'passport', or 'unknown'\n"
        "- 'country': full lowercase country name (like 'united kingdom'), or null if not mentioned\n\n"
        "IMPORTANT:\n"
        "- Only reply with a valid lowercase JSON object\n"
        "- No markdown, no code, no explanation\n"
        "- All keys and values must be in lowercase\n"
        "- Null should be written as: null (without quotes)\n\n"
        "Format:\n"
        '{"intent": "visa", "country": "united kingdom"}\n'
    )

    try:
        response = requests.post("http://localhost:11434/api/chat", json={
            "model": "gemma:2b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }, timeout=5)

        raw_output = ""
        try:
            raw_output = response.json().get("message", {}).get("content", "").strip()
        except Exception:
            raw_output = response.text.strip()

        if raw_output.startswith("```"):
            raw_output = raw_output.strip("```").strip()
            raw_output = raw_output.replace("json", "").replace("python", "").strip()

        if not raw_output:
            return {"intent": "unknown", "country": None}

        try:
            result = json.loads(raw_output)
        except json.JSONDecodeError:
            # fallback: try to extract simple words
            return {"intent": "unknown", "country": None}

        intent = result.get("intent", "unknown").strip().lower()
        country = result.get("country")
        if intent not in {"visa", "passport", "unknown"}:
            intent = "unknown"

        if isinstance(country, str):
            country = normalize_country_alias(country.strip().lower())
        else:
            country = None

        return {"intent": intent, "country": country}

    except requests.exceptions.RequestException:
        # LLM not available — safe fallback
        return {"intent": "unknown", "country": None}
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

STYLE_CSS = r"""/* style.css */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
  font-family: "Poppins", sans-serif;
}

body {
  background: #e6eef4;
}

.chatbot-toggler {
  position: fixed;
  bottom: 30px;
  right: 35px;
  height: 50px;
  width: 50px;
  display: flex;
  border: none;
  outline: none;
  border-radius: 50%;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  background: #0b3d91;
  transition: all 0.2s ease;
}

.chatbot-toggler span {
  color: #fff;
  position: absolute;
}

body.show-chatbot .chatbot-toggler {
  transform: rotate(90deg);
}

.chatbot-toggler span:last-child,
body.show-chatbot .chatbot-toggler span:first-child {
  opacity: 0;
}

body.show-chatbot .chatbot-toggler span:last-child {
  opacity: 1;
}

.chatbot {
  position: fixed;
  right: 35px;
  bottom: 90px;
  width: 420px;
  background: #fff;
  border-radius: 15px;
  overflow: hidden;
  opacity: 0;
  pointer-events: none;
  transform: scale(0.5);
  transform-origin: bottom right;
  box-shadow: 0 0 128px 0 rgba(0,0,0,0.1), 0 32px 64px -48px rgba(0,0,0,0.5);
  transition: all 0.1s ease;
}

body.show-chatbot .chatbot {
  opacity: 1;
  pointer-events: auto;
  transform: scale(1);
}

.chatbot header {
  padding: 16px 0;
  position: relative;
  text-align: center;
  color: #fff;
  background: #0b3d91;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.chatbot header span.close-btn {
  position: absolute;
  right: 15px;
  top: 50%;
  display: none;
  cursor: pointer;
  transform: translateY(-50%);
}

header h2 {
  font-size: 1.4rem;
}

.chatbot .chatbox {
  overflow-y: auto;
  height: 510px;
  padding: 30px 20px 100px;
}

.chatbox .chat {
  display: flex;
  list-style: none;
}

.chatbox .outgoing {
  margin: 20px 0;
  justify-content: flex-end;
}

.chatbox .incoming .vpa-logo {
  font-weight: bold;
  font-size: 0.9rem;
  color: #0b3d91;
  background: #d6e1f2;
  padding: 6px 10px;
  border-radius: 5px;
  margin: 0 10px 7px 0;
  align-self: flex-end;
  display: flex;
  flex-direction: column;
  align-items: center;
  animation: fly 1.8s ease-in-out forwards;
}

.vpa-text {
  font-weight: 600;
  font-size: 1.1rem;
  color: #0b3d91;
}

.vpa-logo .plane {
  font-size: 1.2rem;
  transform: translateX(0);
  animation: takeoff 1.8s ease-in-out forwards;
  margin-top: -10px;
}

.vpa-logo .curve {
  margin-top: -8px;
  opacity: 1;
  animation: fadeCurve 1.5s ease-in-out forwards;
}

.chatbox .chat p {
  white-space: pre-wrap;
  padding: 12px 16px;
  border-radius: 10px 10px 0 10px;
  max-width: 75%;
  color: #fff;
  font-size: 0.95rem;
  background: #0b3d91;
}

.chatbox .incoming p {
  border-radius: 10px 10px 10px 0;
  color: #000;
  background: #f1f4f7;
}

.chat-input {
  display: flex;
  gap: 5px;
  position: absolute;
  bottom: 0;
  width: 100%;
  background: #fff;
  padding: 3px 20px;
  border-top: 1px solid #ddd;
}

.chat-input textarea {
  height: 55px;
  width: 100%;
  border: none;
  outline: none;
  resize: none;
  max-height: 180px;
  padding: 15px 15px 15px 0;
  font-size: 0.95rem;
}

.chat-input span {
  align-self: flex-end;
  color: #0b3d91;
  cursor: pointer;
  height: 55px;
  display: flex;
  align-items: center;
  visibility: hidden;
  font-size: 1.35rem;
}

.chat-input textarea:valid ~ span {
  visibility: visible;
}

@media (max-width: 490px) {
  .chatbot-toggler {
    right: 20px;
    bottom: 20px;
  }
  .chatbot {
    right: 0;
    bottom: 0;
    height: 100%;
    border-radius: 0;
    width: 100%;
  }
  .chatbot .chatbox {
    height: 90%;
    padding: 25px 15px 100px;
  }
  .chatbot .chat-input {
    padding: 5px 15px;
  }
  .chatbot header span {
    display: block;
  }
}

/* Animations */
@keyframes takeoff {
  0%   { transform: translateX(0); }
  30%  { transform: translateX(5px); }
  60%  { transform: translateX(15px); }
  100% { transform: translateX(40px) rotate(-10deg); }
}

@keyframes fadeCurve {
  0%   { opacity: 1; }
  80%  { opacity: 0.5; }
  100% { opacity: 0; }
}

.bot-logo img {
  width: 42px;
  height: 42px;
  border-radius: 50%;
  object-fit: contain;
  margin: 0 10px 7px 0;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}
/* Loading animation for first bot response */
.loading-msg .dot-animate {
  display: inline-block;
  font-style: italic;
  color: #555;
}

.loading-msg .dots::after {
  content: "";
  display: inline-block;
  width: 1em;
  text-align: left;
  animation: dots 1s steps(3, end) infinite;
}

@keyframes dots {
  0% { content: ""; }
  33% { content: "."; }
  66% { content: ".."; }
  100% { content: "..."; }
}

/* Dot Loader Animation */
.dot-loader {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  margin: 8px 0;
}

.dot-loader .dot {
  width: 10px;
  height: 10px;
  background-color: #0b3d91;
  border-radius: 50%;
  animation: pulse 1.2s infinite ease-in-out;
}

.dot-loader .dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot-loader .dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%, 100% {
    opacity: 0.3;
    transform: scale(0.9);
  }
  50% {
    opacity: 1;
    transform: scale(1.2);
  }
}
"""

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

  const span = document.createElement("span");
  span.className = "material-symbols-outlined";
  span.textContent = type === "incoming" ? "" : "";

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
      btn.onclick = () => window.location.href = opt.url;
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

  li.appendChild(span);
  li.appendChild(p);
  chatbox.appendChild(li);
  chatbox.scrollTop = chatbox.scrollHeight;
};

const showLoadingDots = () => {
  const loadingElem = document.createElement("li");
  loadingElem.className = "chat incoming loading-msg";
  loadingElem.innerHTML = `
    <span class="bot-logo">
      <img src="logo.png" alt="Bot Logo" />
    </span>
    <p class="dot-loader"><span class="dot"></span><span class="dot"></span><span class="dot"></span></p>
  `;
  chatbox.appendChild(loadingElem);
  chatbox.scrollTop = chatbox.scrollHeight;
};

const removeLoadingDots = () => {
  const loadingEl = document.querySelector(".loading-msg");
  if (loadingEl) loadingEl.remove();
};

const isGreetingOnly = (msg) => {
  return GREETING_WORDS.includes(msg.toLowerCase().trim());
};

const sendMessage = async () => {
  const message = inputField.value.trim();
  if (!message) return;

  addMessage(message, "outgoing");
  inputField.value = "";

  let effectiveMessage = message;
  if (message.toLowerCase() === "yes" && last_corrected_country) {
    effectiveMessage = last_corrected_country;
    last_corrected_country = null;
  }

  const isGreeting = isGreetingOnly(message);
  let warmupEl = null;

  if (!hasShownWarmUp && !isGreeting) {
    warmupEl = document.createElement("li");
    warmupEl.className = "chat incoming warmup-msg";
    warmupEl.innerHTML = `<span class="material-symbols-outlined"></span><p>✨ Bringing the best information for you... stay tuned! 😉</p>`;
    chatbox.appendChild(warmupEl);
    chatbox.scrollTop = chatbox.scrollHeight;
    hasShownWarmUp = true;
    await new Promise((resolve) => setTimeout(resolve, 1500)); // Slight delay before actual loading dots
  }

  showLoadingDots();  

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id, message: effectiveMessage })
    });

    const data = await response.json();
    session_id = data.session_id;
    localStorage.setItem("vpbot_session_id", session_id);

    removeLoadingDots();
    if (warmupEl) warmupEl.remove();

    if (data.message && data.message.includes("Did you mean **")) {
      const match = data.message.match(/\*\*(.*?)\*\*/);
      if (match && match[1]) {
        last_corrected_country = match[1];
      }
    }

    addMessage(data.message, "incoming");
  } catch (err) {
    removeLoadingDots();
    if (warmupEl) warmupEl.remove();
    console.error("[❌ Backend Error]", err);
    addMessage("⚠️ Unable to connect to the server. Please try again later.", "incoming");
  }
};

sendBtn.addEventListener("click", sendMessage);
inputField.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

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
    # Serve the logo if present in the same folder as the script
    try:
        with open("logo.png", "rb") as f:
            return Response(f.read(), media_type="image/png")
    except FileNotFoundError:
        return Response(b"", media_type="image/png")  # return empty image (browser will show broken icon)
    except Exception:
        return Response(b"", media_type="image/png")

@app.post("/chat")
def chat_response(input: ChatInput):
    # Core chat handling logic (mirrors previous main.py behavior)
    user_input = (input.message or "").strip()
    session_id = get_or_create_session(input.session_id)
    session = get_session(session_id)
    lower_input = user_input.lower().strip()

    print(f"[Request] session={session_id} message={user_input}")

    # 1. Simple greetings
    if lower_input in GREETINGS:
        return {
            "session_id": session_id,
            "message": "Hi there! 👋 I’m here to assist you with visa or passport-related queries. How can I help?"
        }

    # 1b. Small talk / closing
    if lower_input in {"thanks", "thank you", "bye", "okay", "ok", "k"}:
        return {"session_id": session_id, "message": "You're welcome! 😊 If you need help again, just type your query."}

    # 2. Explicit session start
    if (not input.session_id or not str(input.session_id).strip()) and lower_input in ["", "start", "get started"]:
        return {"session_id": session_id, "message": "Hello! 👋 How can I assist you today with your visa or passport?"}

    # 3. Passport quick response
    if "passport" in lower_input:
        update_doc_type(session_id, "passport")
        return {
            "session_id": session_id,
            "message": (
                "Here are a few common passport-related options:\n\n"
                "- [New Passport or First Time Passport](https://www.myvisapassport.com/new-us-passport/)\n"
                "- [Damaged/Lost/Stolen Passport](https://www.myvisapassport.com/passport_stolen/)\n"
                "- [Renewal](https://www.myvisapassport.com/passport-renewal/)\n"
                "- [Second Passport](https://www.myvisapassport.com/second-passport/)\n\n"
                "For other queries, visit: https://www.myvisapassport.com/passport/"
            )
        }

    # 4. Fast visa match if message contains country names directly
    if "visa" in lower_input:
        for country_key in list(country_links.keys()):
            if country_key in lower_input:
                if not is_supported_country(country_key):
                    reset_session(session_id)
                    return {
                        "session_id": session_id,
                        "message": f"Sorry, we currently do not process visa services for {country_key.title()}. Please contact the nearest embassy or consulate for assistance."
                    }
                link = get_link(country_key, "visa")
                if not link:
                    reset_session(session_id)
                    return {
                        "session_id": session_id,
                        "message": f"Sorry, we couldn't find a visa link for {country_key.title()}. Please try again later."
                    }
                reset_session(session_id)
                msg = (f"Great! Here's the link to apply for a visa to {country_key.title()}: {link}\n\n"
                       f"For further help, contact customer service at 1-866-376-1125 or email info@etsonweb.com")
                return {"session_id": session_id, "message": msg}

    # 5. Use LLM to extract intent/country if needed
    result = query_intent_and_country(user_input)
    intent = result.get("intent")
    country_raw = result.get("country")
    print(f"[LLM] intent={intent} country_raw={country_raw}")

    # If LLM says visa + country found
    if intent == "visa" and country_raw:
        country_raw = normalize_country_alias(country_raw)
        country = country_raw if country_raw in country_links else get_closest_country_name(country_raw)
        print(f"[Normalized] country_raw={country_raw} -> country={country}")

        if was_corrected(country_raw, country):
            # Ask user to confirm corrected country
            update_doc_type(session_id, "visa")
            return {
                "session_id": session_id,
                "message": f"Did you mean **{country.title()}**? Please reply with 'yes' to confirm or type the correct country name."
            }

        if not is_supported_country(country):
            reset_session(session_id)
            return {
                "session_id": session_id,
                "message": f"Sorry, we currently do not process visa services for {country.title()}. Please contact the nearest embassy or consulate."
            }

        link = get_link(country, "visa")
        reset_session(session_id)
        if link:
            msg = (f"Great! Here's the link to apply for a visa to {country.title()}: {link}\n\n"
                   f"For further help, contact customer service at 1-866-376-1125 or email info@etsonweb.com")
            return {"session_id": session_id, "message": msg}
        else:
            return {"session_id": session_id, "message": f"Sorry, I couldn't find a visa link for {country.title()}."}

    # 6. If document type not set, ask user
    if not session.get("doc_type"):
        if intent == "passport":
            update_doc_type(session_id, "passport")
            return {
                "session_id": session_id,
                "message": (
                    "Here are a few common passport-related options:\n\n"
                    "- [New Passport or First Time Passport](https://www.myvisapassport.com/new-us-passport/)\n"
                    "- [Damaged/Lost/Stolen Passport](https://www.myvisapassport.com/passport_stolen/)\n"
                    "- [Renewal](https://www.myvisapassport.com/passport-renewal/)\n"
                    "- [Second Passport](https://www.myvisapassport.com/second-passport/)\n\n"
                    "For other queries, visit: https://www.myvisapassport.com/passport/"
                )
            }
        elif intent == "visa":
            update_doc_type(session_id, "visa")
            return {"session_id": session_id, "message": "Which country are you applying for the visa to?"}
        else:
            return {"session_id": session_id, "message": "Hello — are you looking for a visa or a passport? Please tell me which one."}

    # 7. Final fallback: treat user input as country name after doc_type is set
    user_country_input = normalize_country_alias(user_input.strip().lower())
    country = user_country_input if user_country_input in country_links else get_closest_country_name(user_input)
    print(f"[Fallback] interpreted country: {country}")

    if was_corrected(user_input, country):
        update_doc_type(session_id, "visa")
        return {
            "session_id": session_id,
            "message": f"Did you mean **{country.title()}**? Please reply with 'yes' to confirm or type the correct country name."
        }

    if not is_supported_country(country):
        reset_session(session_id)
        return {
            "session_id": session_id,
            "message": (
                f"Sorry, we currently do not process visa services for {country.title()}. "
                "Please contact the nearest embassy or consulate for further assistance."
            )
        }

    link = get_link(country, session.get("doc_type", "visa"))
    reset_session(session_id)
    if link:
        return {
            "session_id": session_id,
            "message": (
                f"Great! Here's the link to apply for a {session.get('doc_type', 'visa')} to {country.title()}: {link}\n\n"
                "For further help, contact customer service at 1-866-376-1125 or email info@etsonweb.com"
            )
        }
    else:
        return {"session_id": session_id, "message": f"Sorry, I couldn't find a {session.get('doc_type','document')} link for {country.title()}."}

# =====================
# MAIN (run server)
# =====================
if __name__ == "__main__":
    import uvicorn, webbrowser, threading, time

    url = "http://localhost:8000"

    # Open browser after short delay so server is ready
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser).start()

    print(f"🚀 VP Bot is running! Open {url} in your browser.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
