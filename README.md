# ThreadModerator 💬

> Next-Generation Conversational Intelligence & WhatsApp Chat Analytics powered by Machine Learning, NLP, and Google Gemini.

ThreadModerator transforms exported WhatsApp conversations into actionable behavioral insights, sentiment trajectories, topic models, and interactive social interaction networks — wrapped in a modern, adaptive UI that seamlessly supports both **Light** and **Dark** modes.

---

## 🚀 Key Capabilities

### 1. 🧠 Sentiment & Mood Intelligence
- **VADER Sentiment Polarity**: Fine-tuned with conversational slang, Hinglish expressions, and emoji sentiment scoring.
- **Sentiment Trajectory**: 7-day rolling mood average tracking group atmosphere over time.
- **Participant Polarity Leaderboard**: Highlights most positive contributors and critical/issue-raising members.

### 2. 🎯 Machine Learning Topics & Intent Classifier
- **Conversational Intent Classification**: Categorizes messages into *Questions & Inquiries*, *Announcements*, *Agreements (+1)*, *Issues & Blockers*, *Action Requests*, and *General Discussions*.
- **Unsupervised Topic Modeling (LDA)**: Uncovers latent themes using Latent Dirichlet Allocation (`scikit-learn`).
- **TF-IDF N-Grams**: Extracts salient 2-gram and 3-gram keyphrases.

### 3. 👥 Participant Dynamics & Behavioral ML
- **Response Latency**: Computes median turn-around response times across participants.
- **Thread Initiators**: Detects members who break communication lulls (>3 hours of silence).
- **Activity Chronotypes**: Classifies activity into *Early Birds*, *Day Active*, *Evening*, and *Night Owls*.

### 4. 🕸️ Social Interaction Network Graph
- **Network Analysis (`NetworkX`)**: Maps who replies to whom within sequential conversational windows.
- **Centrality Metrics**: Interactive 2D Plotly graph sizing nodes by conversational centrality and connection density.

### 5. 🤖 Gemini AI Studio
- **Executive Briefing**: One-click GenAI synthesis of key discussions, decisions, blockers, and next steps.
- **Ask AI ("Chat with your Chat")**: Interactive Q&A over the chat transcript with grounded citations.
- **Participant Personas**: AI-generated psychological archetypes for top contributors.
- *Includes automatic offline heuristic fallbacks when running without an API key.*

### 6. 🎨 Clean, Human-Crafted Adaptive UI
- Supports both **Light Mode** and **Dark Mode** seamlessly via CSS custom properties.
- 100% interactive **Plotly** visualizations.
- Transparent RGBA WordClouds that blend into any theme.
- Out-of-the-box non-confidential sample dataset (`Sample_Team_Chat.txt`).

---

## 🛠️ Quick Start

### 1. Clone & Navigate
```bash
cd ThreadModerator
```

### 2. Setup Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📂 Exporting WhatsApp Chats

1. Open any WhatsApp chat or group.
2. Tap **More Options (⋮)** > **More** > **Export Chat**.
3. Choose **Without Media**.
4. Upload the generated `.txt` file via the sidebar in ThreadModerator.

---

## 🔒 Privacy & Confidentiality
- All processing is conducted locally in memory.
- Confidential `.txt` exports and chat logs are excluded from version control via `.gitignore`.
- Fictional dummy dataset included for safe public demonstrations and testing.
