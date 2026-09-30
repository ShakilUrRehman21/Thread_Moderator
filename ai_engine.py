import re
import os
import numpy as np
import pandas as pd
from collections import Counter
import networkx as nx
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Initialize NLTK VADER Analyzer with auto-download for deployment
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

# Enhance VADER lexicon with modern chat & Hinglish slang + emojis
custom_lexicon = {
    # Positive / Agreement
    '+1': 0.8,
    'agree': 0.6,
    'thanks': 0.8,
    'thank': 0.8,
    'thx': 0.6,
    'ty': 0.5,
    'congrats': 1.0,
    'congratulations': 1.2,
    'mast': 0.9,
    'badhiya': 0.9,
    'sahi': 0.7,
    'shandar': 1.0,
    'op': 0.8,
    'gg': 0.6,
    'awesome': 1.0,
    'great': 0.8,
    'superb': 1.0,
    'fire': 0.7,
    'cool': 0.5,
    'solved': 0.7,
    'helpful': 0.8,
    # Negative / Complaints / Errors
    'fake': -1.2,
    'fraud': -1.5,
    'scam': -1.5,
    'galat': -1.0,
    'wrong': -1.0,
    'error': -0.9,
    'issue': -0.7,
    'problem': -0.8,
    'fail': -1.0,
    'failed': -1.0,
    'bug': -0.8,
    'kharab': -0.9,
    'bekar': -1.0,
    'cheat': -1.3,
    'cheat': -1.3,
    'pain': -0.8,
    'sad': -0.9,
    'stress': -1.0,
    'tension': -0.8,
    # Emojis sentiment
    '😂': 0.8,
    '🤣': 0.9,
    '❤️': 1.0,
    '🔥': 0.8,
    '👍': 0.6,
    '💯': 0.9,
    '🎉': 1.0,
    '🙌': 0.7,
    '😭': -0.6,
    '😢': -0.7,
    '😡': -1.2,
    '🤬': -1.5,
    '💀': -0.3,
    '🥲': -0.2,
    '🙏': 0.6
}
sia.lexicon.update(custom_lexicon)

# Load Stopwords
_STOPWORDS_CACHE = None

def get_stopwords() -> set:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        stopwords = set()
        # English NLTK stopwords with auto-download for deployment
        try:
            from nltk.corpus import stopwords as nltk_stopwords
            stopwords.update(nltk_stopwords.words('english'))
        except (LookupError, AttributeError):
            try:
                import nltk
                nltk.download('stopwords', quiet=True)
                from nltk.corpus import stopwords as nltk_stopwords
                stopwords.update(nltk_stopwords.words('english'))
            except Exception:
                pass
        except Exception:
            pass

        # Local custom Hinglish + chat stop words
        stopwords_file = os.path.join(os.path.dirname(__file__), 'stopWords.txt')
        if os.path.exists(stopwords_file):
            try:
                with open(stopwords_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_words = f.read().split()
                    stopwords.update([w.strip().lower() for w in file_words if len(w.strip()) > 1])
            except Exception:
                pass
        
        # Extra chat tokens to ignore
        stopwords.update({
            'media', 'omitted', 'message', 'deleted', 'pm', 'am', 'https', 'http', 'www',
            'tap', 'learn', 'call', 'end', 'encrypted', 'invite', 'joined', 'link',
            'group', 'changed', 'notification', 'added', 'removed', 'left'
        })
        _STOPWORDS_CACHE = stopwords
    return _STOPWORDS_CACHE


# -------------------------------------------------------------
# 1. Sentiment & Mood Intelligence
# -------------------------------------------------------------
def analyze_sentiment(df: pd.DataFrame, selected_user: str = "Overall") -> dict:
    """
    Computes VADER sentiment polarity and categorizes messages.
    Returns:
    - enriched DataFrame with sentiment_score & sentiment_label
    - summary dict (counts, percentages, average sentiment)
    - daily sentiment timeline
    - user sentiment leaderboard
    """
    data = df.copy()
    if selected_user != "Overall":
        data = data[data['user'] == selected_user]

    # Exclude system notifications and media tags for sentiment
    valid_mask = (~data['is_system']) & (~data['is_media']) & (~data['is_deleted'])
    valid_data = data[valid_mask].copy()

    if valid_data.empty:
        return {
            'df': data,
            'summary': {'Positive': 0, 'Neutral': 0, 'Negative': 0, 'avg_score': 0.0},
            'daily_timeline': pd.DataFrame(),
            'user_rankings': pd.DataFrame()
        }

    scores = []
    labels = []
    for msg in valid_data['message']:
        res = sia.polarity_scores(str(msg))
        comp = res['compound']
        scores.append(comp)
        if comp >= 0.05:
            labels.append('Positive')
        elif comp <= -0.05:
            labels.append('Negative')
        else:
            labels.append('Neutral')

    valid_data['sentiment_score'] = scores
    valid_data['sentiment_label'] = labels

    # Merge back to data
    data['sentiment_score'] = np.nan
    data['sentiment_label'] = 'Neutral'
    data.loc[valid_data.index, 'sentiment_score'] = valid_data['sentiment_score']
    data.loc[valid_data.index, 'sentiment_label'] = valid_data['sentiment_label']

    # Summary
    counts = valid_data['sentiment_label'].value_counts().to_dict()
    total = len(valid_data)
    pos_cnt = counts.get('Positive', 0)
    neu_cnt = counts.get('Neutral', 0)
    neg_cnt = counts.get('Negative', 0)
    avg_score = round(float(valid_data['sentiment_score'].mean()), 3) if total > 0 else 0.0

    summary = {
        'total_analyzed': total,
        'Positive': pos_cnt,
        'Neutral': neu_cnt,
        'Negative': neg_cnt,
        'pos_pct': round((pos_cnt / total) * 100, 1) if total else 0,
        'neu_pct': round((neu_cnt / total) * 100, 1) if total else 0,
        'neg_pct': round((neg_cnt / total) * 100, 1) if total else 0,
        'avg_score': avg_score
    }

    # Daily sentiment timeline
    daily = valid_data.groupby('only_date').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        message_count=('sentiment_score', 'count'),
        pos_count=('sentiment_label', lambda s: (s == 'Positive').sum()),
        neg_count=('sentiment_label', lambda s: (s == 'Negative').sum())
    ).reset_index()
    daily['date'] = pd.to_datetime(daily['only_date'])
    daily.sort_values('date', inplace=True)
    daily['rolling_sentiment'] = daily['avg_sentiment'].rolling(window=7, min_periods=1).mean()

    # User rankings (only users with >= 4 messages)
    user_rankings = pd.DataFrame()
    if selected_user == "Overall":
        user_stats = valid_data.groupby('user').agg(
            msg_count=('sentiment_score', 'count'),
            avg_sentiment=('sentiment_score', 'mean'),
            pos_ratio=('sentiment_label', lambda s: (s == 'Positive').mean() * 100),
            neg_ratio=('sentiment_label', lambda s: (s == 'Negative').mean() * 100)
        ).reset_index()
        user_stats = user_stats[user_stats['msg_count'] >= 4]
        user_stats.sort_values('avg_sentiment', ascending=False, inplace=True)
        user_rankings = user_stats

    return {
        'df': data,
        'summary': summary,
        'daily_timeline': daily,
        'user_rankings': user_rankings
    }


# -------------------------------------------------------------
# 2. Intent & Conversational Dynamics Classification
# -------------------------------------------------------------
def classify_intents(df: pd.DataFrame, selected_user: str = "Overall") -> dict:
    """
    Classifies messages into conversational intents:
    - Question / Help Request
    - Announcement / Notice
    - Agreement / Endorsement (+1)
    - Issue / Complaint
    - Humor & Banter
    - Action Request
    - General Discussion
    """
    data = df.copy()
    if selected_user != "Overall":
        data = data[data['user'] == selected_user]

    valid_data = data[(~data['is_system']) & (~data['is_deleted'])].copy()
    if valid_data.empty:
        return {'counts': {}, 'df': data, 'user_intent': pd.DataFrame()}

    intents = []
    for msg in valid_data['message']:
        m_lower = str(msg).lower().strip()
        
        # 1. Agreement (+1, haan, same, etc)
        if re.search(r'(\+1|^\s*yes\b|^\s*haan\b|^\s*ok\b|^\s*same\b|agree|sahi hai|bilkul|done\b|👍)', m_lower):
            intents.append('Agreement (+1)')
        # 2. Humor & Banter
        elif re.search(r'(😂|🤣|lol|haha|lmao|xd|😆|😅|moye moye)', m_lower):
            intents.append('Humor & Banter')
        # 3. Issue & Complaint
        elif re.search(r'(wrong|galat|issue|problem|fake|fraud|scam|error|fail|kharab|bekar|nahi ho raha|not working)', m_lower):
            intents.append('Issue & Complaint')
        # 4. Question & Inquiry
        elif '?' in m_lower or re.search(r'\b(kya|kaise|how|why|when|where|which|anyone|kaha|batao|kisi ko|doubt|help)\b', m_lower):
            intents.append('Question & Inquiry')
        # 5. Announcement & Information
        elif re.search(r'\b(all students|notice|important|deadline|registration|batch|attend|instruction|announcement)\b', m_lower):
            intents.append('Announcement & Notice')
        # 6. Action Request
        elif re.search(r'\b(fill|submit|verify|contact|forward|bhejo|check|update|apply)\b', m_lower):
            intents.append('Action Request')
        # 7. Media
        elif '<media omitted>' in m_lower:
            intents.append('Media Shared')
        else:
            intents.append('General Discussion')

    valid_data['intent'] = intents
    data['intent'] = 'System'
    data.loc[valid_data.index, 'intent'] = valid_data['intent']

    counts = valid_data['intent'].value_counts().to_dict()

    # User Intent Matrix
    user_intent = pd.DataFrame()
    if selected_user == "Overall":
        top_users = valid_data['user'].value_counts().head(8).index.tolist()
        sub = valid_data[valid_data['user'].isin(top_users)]
        if not sub.empty:
            user_intent = pd.crosstab(sub['user'], sub['intent'], normalize='index') * 100

    return {
        'counts': counts,
        'df': data,
        'user_intent': user_intent
    }


# -------------------------------------------------------------
# 3. Topic Modeling (Unsupervised ML via LDA)
# -------------------------------------------------------------
def extract_topics(df: pd.DataFrame, selected_user: str = "Overall", n_topics: int = 4, top_words_count: int = 6) -> list:
    """
    Applies Latent Dirichlet Allocation (LDA) to extract conversational themes.
    Returns list of dicts: [{'topic_id': 1, 'keywords': [('word', weight), ...], 'summary_label': '...'}]
    """
    data = df.copy()
    if selected_user != "Overall":
        data = data[data['user'] == selected_user]

    valid_msgs = data[(~data['is_system']) & (~data['is_media']) & (~data['is_deleted'])]['message'].dropna().tolist()
    
    # Filter short messages (< 3 words)
    valid_msgs = [m for m in valid_msgs if len(m.split()) >= 3]
    if len(valid_msgs) < 15:
        return []

    stops = list(get_stopwords())
    vectorizer = CountVectorizer(
        stop_words=stops,
        min_df=2,
        max_df=0.85,
        max_features=800,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    try:
        X = vectorizer.fit_transform(valid_msgs)
        if X.shape[1] < n_topics:
            return []
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
        lda.fit(X)

        feature_names = vectorizer.get_feature_names_out()
        topics = []
        for idx, comp in enumerate(lda.components_):
            top_indices = comp.argsort()[:-top_words_count - 1:-1]
            words_weights = [(feature_names[i], round(float(comp[i]), 2)) for i in top_indices]
            words_only = [w[0] for w in words_weights]
            topics.append({
                'topic_id': idx + 1,
                'name': f"Topic {idx + 1}: {words_only[0].capitalize()} & {words_only[1].capitalize()}",
                'keywords': words_weights,
                'words_string': ", ".join(words_only)
            })
        return topics
    except Exception:
        return []


# -------------------------------------------------------------
# 4. Semantic Keyphrases (TF-IDF N-Grams)
# -------------------------------------------------------------
def extract_keyphrases(df: pd.DataFrame, selected_user: str = "Overall", top_n: int = 15) -> pd.DataFrame:
    """
    Extracts high-impact 2-gram and 3-gram keyphrases using TF-IDF.
    """
    data = df.copy()
    if selected_user != "Overall":
        data = data[data['user'] == selected_user]

    valid_msgs = data[(~data['is_system']) & (~data['is_media']) & (~data['is_deleted'])]['message'].dropna().tolist()
    valid_msgs = [m for m in valid_msgs if len(m.split()) >= 2]
    if len(valid_msgs) < 10:
        return pd.DataFrame()

    stops = list(get_stopwords())
    tfidf = TfidfVectorizer(
        ngram_range=(2, 3),
        stop_words=stops,
        min_df=2,
        max_features=top_n * 2,
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b'
    )
    try:
        X = tfidf.fit_transform(valid_msgs)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = tfidf.get_feature_names_out()
        phrase_df = pd.DataFrame({'phrase': features, 'score': scores})
        phrase_df.sort_values('score', ascending=False, inplace=True)
        phrase_df['score'] = phrase_df['score'].round(2)
        return phrase_df.head(top_n)
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------
# 5. Social Interaction Network Graph (Graph ML)
# -------------------------------------------------------------
def build_interaction_network(df: pd.DataFrame, max_time_gap_mins: int = 10, max_users: int = 20) -> dict:
    """
    Constructs a directed social interaction network from conversational sequence.
    Nodes = Users, Edges = Conversational replies within time threshold.
    """
    data = df[(~df['is_system']) & (~df['is_deleted'])].sort_values('date').reset_index(drop=True)
    if data.empty:
        return {'nodes': [], 'edges': [], 'centrality': {}}

    # Filter to top active users to keep visualization crisp
    top_users = set(data['user'].value_counts().head(max_users).index.tolist())
    filtered_data = data[data['user'].isin(top_users)].reset_index(drop=True)

    G = nx.DiGraph()
    prev_user = None
    prev_time = None

    for _, row in filtered_data.iterrows():
        curr_user = row['user']
        curr_time = row['date']
        if prev_user and prev_user != curr_user:
            delta_mins = (curr_time - prev_time).total_seconds() / 60.0
            if delta_mins <= max_time_gap_mins:
                if G.has_edge(prev_user, curr_user):
                    G[prev_user][curr_user]['weight'] += 1
                else:
                    G.add_edge(prev_user, curr_user, weight=1)
        prev_user = curr_user
        prev_time = curr_time

    if G.number_of_nodes() == 0:
        return {'nodes': [], 'edges': [], 'centrality': {}}

    # Compute network metrics
    in_deg = dict(G.in_degree(weight='weight'))
    out_deg = dict(G.out_degree(weight='weight'))
    degree_cent = nx.degree_centrality(G)

    # 2D Spring Layout positions
    pos = nx.spring_layout(G, k=0.5, iterations=40, seed=42)

    nodes_list = []
    for node in G.nodes():
        x, y = pos[node]
        nodes_list.append({
            'id': node,
            'name': node,
            'x': float(x),
            'y': float(y),
            'in_degree': in_deg.get(node, 0),
            'out_degree': out_deg.get(node, 0),
            'total_degree': in_deg.get(node, 0) + out_deg.get(node, 0),
            'centrality': round(degree_cent.get(node, 0), 3)
        })

    edges_list = []
    for u, v, attrs in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edges_list.append({
            'source': u,
            'target': v,
            'weight': attrs['weight'],
            'x0': float(x0),
            'y0': float(y0),
            'x1': float(x1),
            'y1': float(y1)
        })

    return {
        'nodes': nodes_list,
        'edges': edges_list,
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges()
    }


# -------------------------------------------------------------
# 6. Behavioral ML: Response Latency & Chronotype Profiling
# -------------------------------------------------------------
def analyze_behavioral_dynamics(df: pd.DataFrame) -> dict:
    """
    Analyzes:
    - Average response latency per user (in minutes)
    - Conversation initiators (who breaks a silence > 3 hours)
    - Chronotype distribution (Early Bird vs Day Active vs Night Owl)
    """
    data = df[(~df['is_system'])].sort_values('date').reset_index(drop=True)
    if data.empty:
        return {}

    # 1. Response latency
    latencies = {}
    conversation_starters = Counter()
    prev_user = None
    prev_time = None

    for _, row in data.iterrows():
        curr_user = row['user']
        curr_time = row['date']
        if prev_time is not None:
            delta_mins = (curr_time - prev_time).total_seconds() / 60.0
            if delta_mins > 180:  # 3 hour lull -> conversation starter
                conversation_starters[curr_user] += 1
            elif prev_user != curr_user and delta_mins < 60:
                if curr_user not in latencies:
                    latencies[curr_user] = []
                latencies[curr_user].append(delta_mins)
        prev_user = curr_user
        prev_time = curr_time

    avg_latency = {}
    for user, times in latencies.items():
        if len(times) >= 3:
            avg_latency[user] = round(float(np.median(times)), 1)

    latency_df = pd.DataFrame(list(avg_latency.items()), columns=['user', 'median_reply_mins'])
    latency_df.sort_values('median_reply_mins', inplace=True)

    starter_df = pd.DataFrame(conversation_starters.most_common(10), columns=['user', 'threads_started'])

    # 2. Chronotype profiling
    # Night Owl: 22:00 to 05:00
    # Early Bird: 05:00 to 10:00
    # Day Worker: 10:00 to 18:00
    # Evening: 18:00 to 22:00
    def get_chronotype(hour):
        if hour >= 22 or hour < 5:
            return 'Night Owl 🦉'
        elif 5 <= hour < 10:
            return 'Early Bird 🌅'
        elif 10 <= hour < 18:
            return 'Day Active ☀️'
        else:
            return 'Evening Social 🌆'

    data['chronotype'] = data['hour'].apply(get_chronotype)
    chrono_counts = data['chronotype'].value_counts().to_dict()

    return {
        'latency_df': latency_df,
        'starter_df': starter_df,
        'chronotype_counts': chrono_counts
    }


# -------------------------------------------------------------
# 7. Gemini AI Studio (Google GenAI Integration)
# -------------------------------------------------------------
def get_gemini_client(api_key: str = None):
    """Initializes Google GenAI Client with key from argument or environment."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception:
        return None


def generate_gemini_summary(df: pd.DataFrame, api_key: str = None) -> str:
    """
    Generates an executive briefing using Gemini 2.5 Flash,
    with an intelligent extractive fallback if no API key is provided.
    """
    client = get_gemini_client(api_key)
    
    # Prepare representative context snippet
    valid_msgs = df[(~df['is_system']) & (~df['is_deleted'])].copy()
    total_msgs = len(valid_msgs)
    total_users = valid_msgs['user'].nunique()
    start_date = valid_msgs['date'].min().strftime('%d %b %Y') if not valid_msgs.empty else 'N/A'
    end_date = valid_msgs['date'].max().strftime('%d %b %Y') if not valid_msgs.empty else 'N/A'

    # Sample top messages + discussion bursts
    sample_msgs = []
    # Sample every Nth message to cover entire span up to 100 messages
    step = max(1, total_msgs // 80)
    for idx, row in valid_msgs.iloc[::step].head(80).iterrows():
        sample_msgs.append(f"[{row['date'].strftime('%Y-%m-%d %H:%M')}] {row['user']}: {row['message'][:120]}")
    chat_sample_text = "\n".join(sample_msgs)

    if client:
        try:
            prompt = f"""You are an elite conversation intelligence analyst. Analyze this WhatsApp group chat and generate a structured, executive-level report.

CHAT METRICS:
- Total Messages: {total_msgs}
- Total Participants: {total_users}
- Date Range: {start_date} to {end_date}

REPRESENTATIVE CHAT TRANSCRIPT:
{chat_sample_text}

Provide your response in beautifully formatted GitHub Markdown with these exact sections:
### 🎯 Executive Synopsis
(A crisp 2-3 paragraph summary of what this chat group is about and the primary purpose/goals discussed)

### 📌 Key Highlights & Major Topics
(Bullet points detailing the most significant announcements, debates, or questions raised)

### ⚠️ Pain Points & Unresolved Issues
(Specific confusion, complaints, marks discrepancies, or recurring hurdles encountered by members)

### 🤝 Group Dynamic & Atmospheric Tone
(Assessment of collaboration, responsiveness, sentiment, and member engagement)

### 📋 Action Items & Takeaways
(Actionable items for the participants or organizers)
"""
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"*(Gemini API encountered an issue: {str(e)}. Displaying heuristic briefing below)*\n\n" + _fallback_heuristic_summary(df)
    else:
        return _fallback_heuristic_summary(df)


def ask_gemini_chat(query: str, df: pd.DataFrame, api_key: str = None) -> str:
    """
    RAG-style conversational Q&A over the chat dataset.
    """
    client = get_gemini_client(api_key)
    if not client:
        return "⚠️ **Gemini API Key Required**: Please provide your Google Gemini API Key in the sidebar to activate the AI Chat Assistant."

    valid_msgs = df[(~df['is_system']) & (~df['is_deleted'])].copy()
    
    # Filter messages relevant to query keywords
    query_tokens = [w.lower() for w in re.findall(r'\b\w{3,}\b', query) if w.lower() not in get_stopwords()]
    matching_rows = []
    
    if query_tokens:
        for idx, row in valid_msgs.iterrows():
            m_lower = str(row['message']).lower()
            u_lower = str(row['user']).lower()
            if any(t in m_lower or t in u_lower for t in query_tokens):
                matching_rows.append(row)
    
    # If matches found, take top 40 matching rows; otherwise take uniform sample of 60 rows
    if len(matching_rows) >= 5:
        context_df = pd.DataFrame(matching_rows).head(50)
    else:
        step = max(1, len(valid_msgs) // 60)
        context_df = valid_msgs.iloc[::step].head(60)

    context_lines = []
    for _, row in context_df.iterrows():
        context_lines.append(f"[{row['date'].strftime('%d/%m/%y %H:%M')}] {row['user']}: {row['message']}")
    context_text = "\n".join(context_lines)

    prompt = f"""You are an AI assistant who answers questions strictly based on the following WhatsApp chat transcript.
Cite user names, dates, or exact message snippets whenever relevant. If the answer is not mentioned in the transcript, state that clearly.

CHAT TRANSCRIPT EXCERPT:
{context_text}

USER QUESTION: {query}

Answer concisely, helpfully, and clearly in markdown:"""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Error querying Gemini API: {str(e)}"


def generate_user_personas(df: pd.DataFrame, api_key: str = None) -> list:
    """
    Generates AI persona profiles for the top 4 active participants.
    """
    valid_msgs = df[(~df['is_system']) & (~df['is_deleted'])].copy()
    top_users = valid_msgs['user'].value_counts().head(4).index.tolist()
    
    personas = []
    client = get_gemini_client(api_key)

    for user in top_users:
        u_msgs = valid_msgs[valid_msgs['user'] == user]
        count = len(u_msgs)
        avg_words = round(u_msgs['word_count'].mean(), 1)
        sample = u_msgs['message'].head(15).tolist()
        sample_str = " | ".join(str(s)[:60] for s in sample)

        if client:
            try:
                p_prompt = f"""Analyze these messages from user '{user}' in a WhatsApp group:
Sample messages: {sample_str}
Total messages: {count}, Avg words per message: {avg_words}.

Generate a fun, insightful personality card with:
1. Archetype Title (e.g., 'The Resource Anchor', 'The Inquisitive Solver', 'The Silent Approver')
2. 2-sentence Personality Description
3. Characteristic Catchphrase / Vibe
4. Badges (e.g. 🎯 Task-Driven, ⚡ Rapid Responder, 🦉 Night Thinker)

Format as JSON with keys: title, description, catchphrase, badges (array). Return ONLY valid JSON."""
                res = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=p_prompt
                )
                import json
                clean_json = re.search(r'\{.*\}', res.text, re.DOTALL)
                if clean_json:
                    p_data = json.loads(clean_json.group(0))
                    personas.append({
                        'user': user,
                        'title': p_data.get('title', 'Group Contributor'),
                        'description': p_data.get('description', 'Active member of the chat.'),
                        'catchphrase': p_data.get('catchphrase', 'Let us connect on this.'),
                        'badges': p_data.get('badges', ['Active Member'])
                    })
                    continue
            except Exception:
                pass

        # Offline heuristic persona fallback
        has_questions = any('?' in str(m) for m in sample)
        has_plus_one = any('+1' in str(m) for m in sample)
        has_links = any('http' in str(m) for m in sample)

        if has_links or avg_words > 12:
            title = "The Information Hub 📡"
            desc = f"Maintains high initiative with detailed updates and informative messages (avg {avg_words} words)."
            badges = ["Informant", "High Context", "Resource Provider"]
        elif has_plus_one:
            title = "The Solidarity Champion 🤝"
            desc = "Always active in seconding important updates, backing up peers, and keeping group momentum alive."
            badges = ["Consensus Builder", "+1 Veteran", "Quick Responder"]
        elif has_questions:
            title = "The Inquisitive Explorer 🔍"
            desc = "Frequently raises pertinent questions, clarifying doubts and uncovering critical details for everyone."
            badges = ["Curious Mind", "Doubt Clarifier", "Active Seeker"]
        else:
            title = "The Group Catalyst ⚡"
            desc = f"A reliable pillar of the conversation with {count} messages driving daily discussions."
            badges = ["Core Contributor", "Consistent Voice", "Pillar"]

        personas.append({
            'user': user,
            'title': title,
            'description': desc,
            'catchphrase': f"Consistently contributing with {count} messages.",
            'badges': badges
        })

    return personas


def _fallback_heuristic_summary(df: pd.DataFrame) -> str:
    """Generates an extractive NLP briefing without external API requirements."""
    valid_msgs = df[(~df['is_system']) & (~df['is_deleted'])].copy()
    if valid_msgs.empty:
        return "No sufficient messages to generate an extractive briefing."

    total_msgs = len(valid_msgs)
    top_users = valid_msgs['user'].value_counts().head(5).to_dict()
    top_phrases = extract_keyphrases(df, top_n=6)
    key_themes = ", ".join(top_phrases['phrase'].tolist()) if not top_phrases.empty else "General discussions"
    
    peak_date = valid_msgs['only_date'].value_counts().idxmax()
    peak_count = valid_msgs['only_date'].value_counts().max()
    
    # Sample questions
    questions = [m for m in valid_msgs['message'] if '?' in str(m)][:3]
    q_str = "\n".join(f"- *\"{q.strip()}\"*" for q in questions) if questions else "- General updates and confirmations."

    summary_md = f"""### 🎯 Executive Synopsis (Extractive Engine)
This chat contains **{total_msgs:,} active messages** shared across **{valid_msgs['user'].nunique()} participants**. The conversation displays high collaborative density, with the busiest discussion day recorded on **{peak_date}** with **{peak_count} messages**.

### 📌 Key Highlight Themes
Unsupervised NLP phrase modeling indicates prime focus on:
- **{key_themes}**

### 👥 Top Core Contributors
The primary discussion anchors driving this chat include:
{chr(10).join(f"- **{user}**: {cnt} messages ({round(cnt/total_msgs*100, 1)}% share)" for user, cnt in top_users.items())}

### ❓ Sample Notable Inquiries
{q_str}

### 💡 Pro Tip
*To unlock conversational Q&A and GenAI synthesis, add your free **Google Gemini API Key** in the sidebar settings!*
"""
    return summary_md
