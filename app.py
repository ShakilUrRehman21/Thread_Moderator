import streamlit as st
import pandas as pd
import numpy as np
import os
import io

import preprocessor
import helper
import ai_engine

# -------------------------------------------------------------
# Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="ThreadModerator | Conversational Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# Adaptive Design System (Seamless Light & Dark Theme Support)
# Uses Streamlit CSS variables: --background-color, --secondary-background-color, --text-color
# -------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Metric Cards - Seamless Light & Dark Theme */
    .metric-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 16px 18px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        border-color: rgba(128, 128, 128, 0.4);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .metric-title {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-color);
        opacity: 0.65;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text-color);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.76rem;
        color: var(--text-color);
        opacity: 0.6;
        margin-top: 4px;
    }

    /* Page Header */
    .page-title {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-color);
        margin-bottom: 2px;
    }
    .page-subtitle {
        color: var(--text-color);
        opacity: 0.65;
        font-size: 0.88rem;
        margin-bottom: 20px;
    }

    /* Topic Pill */
    .topic-pill {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-left: 3px solid #10b981;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .topic-name {
        color: var(--text-color);
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Persona Card */
    .persona-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .persona-title {
        color: #0284c7;
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .persona-user {
        font-weight: 600;
        color: var(--text-color);
        font-size: 1rem;
        margin-bottom: 2px;
    }
    .persona-desc {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 0.83rem;
        margin-top: 8px;
        line-height: 1.4;
    }

    .badge {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 500;
        background: rgba(128, 128, 128, 0.12);
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-right: 5px;
        margin-top: 6px;
    }

    /* Segmented Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: transparent;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 14px;
        background-color: transparent;
        color: var(--text-color);
        opacity: 0.7;
        border: none;
        font-size: 0.86rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        opacity: 1 !important;
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------
# Data Ingestion & Caching
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_preprocess(raw_text: str):
    return preprocessor.preprocess(raw_text)


# -------------------------------------------------------------
# Sidebar: Brand, Dataset Selection & Scope
# -------------------------------------------------------------
with st.sidebar:
    # Clean Minimalist Logo
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 20px;">
            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981;"></span>
            <span style="font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;">ThreadModerator</span>
            <span style="font-size: 0.7rem; opacity: 0.7; border: 1px solid rgba(128,128,128,0.3); padding: 2px 6px; border-radius: 4px; margin-left: auto;">v2.4</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Chat Dataset")
    
    # Quick Load Demo Dataset
    if st.button("📁 Load Demo Team Chat", use_container_width=True, help="Load clean fictional sample team chat"):
        st.session_state['active_source'] = "sample_team"

    uploaded_file = st.file_uploader(
        "Upload WhatsApp Chat (.txt)",
        type=["txt"],
        help="Export any WhatsApp chat without media and upload the .txt file."
    )

    if uploaded_file is not None:
        st.session_state['active_source'] = "uploaded"

    st.markdown("---")
    st.markdown("##### Gemini AI Integration")
    user_api_key = st.text_input(
        "Gemini API Key (Optional)",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Free API key from Google AI Studio (aistudio.google.com) to enable conversational Q&A and GenAI synthesis."
    )
    if user_api_key:
        st.caption("Active: Gemini 2.5 Flash connected")
    else:
        st.caption("Active: Offline NLP Heuristics engine")


# -------------------------------------------------------------
# Load Data Content
# -------------------------------------------------------------
raw_data = None
current_source_name = "Demo Team Chat"

active_source = st.session_state.get('active_source', 'sample_team')

if active_source == "uploaded" and uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    raw_data = bytes_data.decode("utf-8", errors="ignore")
    current_source_name = uploaded_file.name
else:
    # Default to non-confidential sample team chat
    sample_path = os.path.join(os.path.dirname(__file__), 'Sample_Team_Chat.txt')
    if os.path.exists(sample_path):
        with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_data = f.read()
        current_source_name = "Sample Team Chat"

if not raw_data:
    st.info("Please upload a WhatsApp chat `.txt` export or load the demo dataset from the sidebar.")
    st.stop()

df = load_and_preprocess(raw_data)
if df.empty:
    st.error("Could not parse messages. Please verify the file format.")
    st.stop()


# -------------------------------------------------------------
# Participant & Date Range Filters
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("##### Filter & Scope")
    
    users = df['user'].unique().tolist()
    if 'group_notification' in users:
        users.remove('group_notification')
    users.sort()
    users.insert(0, "Overall")
    
    selected_user = st.selectbox("Participant", users, index=0)

    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    if min_date < max_date:
        selected_dates = st.slider(
            "Date Range",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="DD/MM/YY"
        )
        df = df[(df['only_date'] >= selected_dates[0]) & (df['only_date'] <= selected_dates[1])].reset_index(drop=True)

    st.markdown("---")
    st.caption(f"Source: **{current_source_name}**")
    st.caption(f"Span: {df['date'].min().strftime('%d %b %Y')} – {df['date'].max().strftime('%d %b %Y')}")


# -------------------------------------------------------------
# Compute Core Stats & AI Insights
# -------------------------------------------------------------
stats = helper.fetch_stats(selected_user, df)
sentiment_res = ai_engine.analyze_sentiment(df, selected_user)
sent_summary = sentiment_res['summary']


# -------------------------------------------------------------
# Header & Summary Cards
# -------------------------------------------------------------
st.markdown('<div class="page-title">Conversation Intelligence</div>', unsafe_allow_html=True)
st.markdown(f'<div class="page-subtitle">Dataset: <b>{current_source_name}</b> &nbsp;•&nbsp; Participant: <span style="color:#10b981; font-weight:600;">{selected_user}</span></div>', unsafe_allow_html=True)

# 6 Clean Metric Cards
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Messages</div>
            <div class="metric-value">{stats['num_messages']:,}</div>
            <div class="metric-sub">~{stats['avg_per_day']} / day</div>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Words</div>
            <div class="metric-value">{stats['total_words']:,}</div>
            <div class="metric-sub">~{round(stats['total_words']/max(1, stats['num_messages']), 1)} words/msg</div>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Participants</div>
            <div class="metric-value">{stats['active_participants']}</div>
            <div class="metric-sub">{stats['days_span']} days active</div>
        </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Media Files</div>
            <div class="metric-value">{stats['num_media']}</div>
            <div class="metric-sub">{round(stats['num_media']/max(1, stats['num_messages'])*100, 1)}% of total</div>
        </div>
    """, unsafe_allow_html=True)

with c5:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Links Shared</div>
            <div class="metric-value">{stats['total_links']}</div>
            <div class="metric-sub">References posted</div>
        </div>
    """, unsafe_allow_html=True)

with c6:
    score = sent_summary.get('avg_score', 0.0)
    sentiment_label = "Positive" if score >= 0.05 else ("Negative" if score <= -0.05 else "Neutral")
    score_color = "#10b981" if score >= 0.05 else ("#f43f5e" if score <= -0.05 else "#64748b")
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sentiment</div>
            <div class="metric-value" style="color: {score_color};">{score:+.2f}</div>
            <div class="metric-sub" style="color: {score_color};">{sentiment_label}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# -------------------------------------------------------------
# Clean Navigation Tabs
# -------------------------------------------------------------
tab_exec, tab_sent, tab_topics, tab_users, tab_network, tab_lexical, tab_gemini = st.tabs([
    "Overview & Velocity",
    "Sentiment Analysis",
    "Topics & Intents",
    "Participant Dynamics",
    "Interaction Graph",
    "Lexical & Emojis",
    "AI Studio & Q&A"
])


# =============================================================
# TAB 1: OVERVIEW & VELOCITY
# =============================================================
with tab_exec:
    st.markdown("#### Message Velocity Over Time")
    daily_df = helper.daily_timeline(selected_user, df)
    if not daily_df.empty:
        fig_time = helper.plot_interactive_timeline(daily_df)
        st.plotly_chart(fig_time, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Activity Heatmap (Day vs Period)")
        heatmap_df = helper.activity_heatmap(selected_user, df)
        fig_heat = helper.plot_heatmap_interactive(heatmap_df)
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_t2:
        st.markdown("#### Activity Distribution")
        day_counts = helper.week_activity_map(selected_user, df)
        month_counts = helper.month_activity_map(selected_user, df)
        
        tab_d, tab_m = st.tabs(["By Day of Week", "By Month"])
        with tab_d:
            import plotly.express as px
            fig_d = px.bar(
                x=day_counts.index, y=day_counts.values,
                labels={'x': 'Day of Week', 'y': 'Messages'},
                color=day_counts.values,
                color_continuous_scale='Blues'
            )
            fig_d.update_layout(margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_d, use_container_width=True)
        with tab_m:
            fig_m = px.bar(
                x=month_counts.index, y=month_counts.values,
                labels={'x': 'Month', 'y': 'Messages'},
                color=month_counts.values,
                color_continuous_scale='Tealgrn'
            )
            fig_m.update_layout(margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_m, use_container_width=True)


# =============================================================
# TAB 2: SENTIMENT ANALYSIS
# =============================================================
with tab_sent:
    st.markdown("#### Sentiment Polarity & Group Atmosphere")
    
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        fig_gauge = helper.plot_sentiment_gauge(sent_summary)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div style="background-color: var(--secondary-background-color); padding:14px; border-radius:8px; border:1px solid rgba(128,128,128,0.2); font-size:0.86rem;">
            <div style="opacity: 0.7; font-weight:600; margin-bottom:8px;">BREAKDOWN</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#10b981;">Positive Messages:</span>
                <b>{sent_summary.get('Positive', 0)} ({sent_summary.get('pos_pct', 0)}%)</b>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="opacity: 0.75;">Neutral Messages:</span>
                <b>{sent_summary.get('Neutral', 0)} ({sent_summary.get('neu_pct', 0)}%)</b>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#f43f5e;">Negative Messages:</span>
                <b>{sent_summary.get('Negative', 0)} ({sent_summary.get('neg_pct', 0)}%)</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        daily_sent = sentiment_res['daily_timeline']
        fig_sent_line = helper.plot_sentiment_trajectory(daily_sent)
        st.plotly_chart(fig_sent_line, use_container_width=True)

    if selected_user == "Overall" and not sentiment_res['user_rankings'].empty:
        st.markdown("#### Participant Polarity Index")
        col_p1, col_p2 = st.columns(2)
        ur = sentiment_res['user_rankings']
        
        with col_p1:
            st.markdown("##### Highest Positive Polarity")
            top_pos = ur.head(5)[['user', 'avg_sentiment', 'msg_count', 'pos_ratio']].copy()
            top_pos.columns = ['Participant', 'Avg Polarity', 'Messages', 'Positive Ratio (%)']
            top_pos['Avg Polarity'] = top_pos['Avg Polarity'].round(3)
            top_pos['Positive Ratio (%)'] = top_pos['Positive Ratio (%)'].round(1)
            st.dataframe(top_pos, use_container_width=True, hide_index=True)

        with col_p2:
            st.markdown("##### Critical / Issue Raising Contributors")
            top_crit = ur.sort_values('avg_sentiment', ascending=True).head(5)[['user', 'avg_sentiment', 'msg_count', 'neg_ratio']].copy()
            top_crit.columns = ['Participant', 'Avg Polarity', 'Messages', 'Critical Ratio (%)']
            top_crit['Avg Polarity'] = top_crit['Avg Polarity'].round(3)
            top_crit['Critical Ratio (%)'] = top_crit['Critical Ratio (%)'].round(1)
            st.dataframe(top_crit, use_container_width=True, hide_index=True)


# =============================================================
# TAB 3: TOPICS & INTENTS
# =============================================================
with tab_topics:
    col_top1, col_top2 = st.columns([1, 1])

    with col_top1:
        st.markdown("#### Conversational Intent Classification")
        intent_data = ai_engine.classify_intents(df, selected_user)
        fig_radar = helper.plot_intent_radar(intent_data['counts'])
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_top2:
        st.markdown("#### Unsupervised Topic Modeling (LDA)")
        topics = ai_engine.extract_topics(df, selected_user, n_topics=4)
        if topics:
            for t in topics:
                keywords_html = "".join([f"<span class='badge'>{w[0]} ({w[1]})</span>" for w in t['keywords']])
                st.markdown(f"""
                <div class="topic-pill">
                    <div class="topic-name">{t['name']}</div>
                    <div style="margin-top: 6px;">{keywords_html}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Insufficient text for distinct LDA topic clustering.")

    st.markdown("#### Salient Keyphrases (TF-IDF N-Grams)")
    keyphrases = ai_engine.extract_keyphrases(df, selected_user, top_n=10)
    if not keyphrases.empty:
        col_kp1, col_kp2 = st.columns([2, 1])
        with col_kp1:
            fig_kp = px.bar(
                keyphrases.iloc[::-1],
                x='score', y='phrase',
                orientation='h',
                labels={'score': 'Relevance Score', 'phrase': ''},
                color='score',
                color_continuous_scale='Blues'
            )
            fig_kp.update_layout(margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_kp, use_container_width=True)
        with col_kp2:
            st.dataframe(keyphrases, use_container_width=True, hide_index=True)


# =============================================================
# TAB 4: PARTICIPANT DYNAMICS & BEHAVIORAL ML
# =============================================================
with tab_users:
    if selected_user == "Overall":
        col_u1, col_u2 = st.columns([1, 1])
        with col_u1:
            st.markdown("#### Most Active Contributors")
            busy_counts, busy_df = helper.most_busy_users(df, top_n=10)
            fig_busy = helper.plot_busy_users_interactive(busy_counts)
            st.plotly_chart(fig_busy, use_container_width=True)

        with col_u2:
            st.markdown("#### Contributor Share")
            st.dataframe(busy_df, use_container_width=True, hide_index=True)

        st.markdown("#### Behavioral Dynamics")
        behavioral = ai_engine.analyze_behavioral_dynamics(df)
        
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.markdown("##### Median Reply Latency")
            st.caption("Median response time (minutes) in conversational turns:")
            if 'latency_df' in behavioral and not behavioral['latency_df'].empty:
                st.dataframe(behavioral['latency_df'].head(8), use_container_width=True, hide_index=True)
            else:
                st.write("Insufficient sequential turn data.")

        with col_b2:
            st.markdown("##### Thread Initiators")
            st.caption("Initiates discussions after a silence (>3 hours):")
            if 'starter_df' in behavioral and not behavioral['starter_df'].empty:
                st.dataframe(behavioral['starter_df'].head(8), use_container_width=True, hide_index=True)
            else:
                st.write("No long silence breaks detected.")

        with col_b3:
            st.markdown("##### Activity Chronotypes")
            st.caption("Time distribution of participant messages:")
            chrono = behavioral.get('chronotype_counts', {})
            if chrono:
                fig_ch = px.pie(
                    names=list(chrono.keys()),
                    values=list(chrono.values()),
                    hole=0.5,
                    color_discrete_sequence=['#6366f1', '#38bdf8', '#10b981', '#f59e0b']
                )
                fig_ch.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_ch, use_container_width=True)
    else:
        st.info(f"Filtered for participant: **{selected_user}**. Switch to 'Overall' in the sidebar to view full group comparative rankings.")


# =============================================================
# TAB 5: INTERACTION GRAPH
# =============================================================
with tab_network:
    st.markdown("#### Social Interaction Network")
    st.caption("Node size reflects conversational centrality; edges denote sequential replies within a 10-minute conversational window.")
    
    net_data = ai_engine.build_interaction_network(df, max_time_gap_mins=10, max_users=25)
    if net_data.get('num_nodes', 0) > 0:
        fig_net = helper.plot_network_interactive(net_data)
        st.plotly_chart(fig_net, use_container_width=True)
        st.caption(f"Network topology: **{net_data['num_nodes']} active nodes**, **{net_data['num_edges']} reply edges**.")
    else:
        st.info("Insufficient interactive multi-turn conversations detected.")


# =============================================================
# TAB 6: LEXICAL & EMOJIS
# =============================================================
with tab_lexical:
    col_l1, col_l2 = st.columns([1, 1])

    with col_l1:
        st.markdown("#### WordCloud")
        wc = helper.create_wordcloud(selected_user, df)
        st.image(wc.to_array(), use_container_width=True)

    with col_l2:
        st.markdown("#### High Frequency Terms")
        word_freq_df = helper.most_common_words(selected_user, df, top_n=15)
        fig_wf = helper.plot_word_frequency_interactive(word_freq_df)
        st.plotly_chart(fig_wf, use_container_width=True)

    st.markdown("#### Emoji Analytics")
    emoji_df = helper.emoji_helper(selected_user, df, top_n=12)
    if not emoji_df.empty:
        col_e1, col_e2 = st.columns([1, 1])
        with col_e1:
            fig_em = helper.plot_emoji_distribution(emoji_df)
            st.plotly_chart(fig_em, use_container_width=True)
        with col_e2:
            st.dataframe(emoji_df, use_container_width=True, hide_index=True)
    else:
        st.info("No emojis found in the selected range.")


# =============================================================
# TAB 7: AI STUDIO & Q&A
# =============================================================
with tab_gemini:
    st.markdown("#### Conversational Intelligence Studio")
    st.caption("Utilizes Google Gemini 2.5 Flash with automatic offline NLP fallbacks.")

    ai_sub1, ai_sub2, ai_sub3 = st.tabs([
        "Executive Briefing",
        "Ask AI (Chat Q&A)",
        "Participant Personas"
    ])

    # 1. Executive Briefing
    with ai_sub1:
        st.markdown("##### Executive Synopsis & Action Items")
        if st.button("Generate Briefing", type="primary"):
            with st.spinner("Synthesizing key discussions..."):
                briefing_text = ai_engine.generate_gemini_summary(df, api_key=user_api_key)
                st.session_state['briefing'] = briefing_text

        if 'briefing' in st.session_state:
            st.markdown(st.session_state['briefing'])
        else:
            briefing_text = ai_engine.generate_gemini_summary(df, api_key=user_api_key)
            st.markdown(briefing_text)

    # 2. Ask AI
    with ai_sub2:
        st.markdown("##### Ask Questions About This Chat")
        
        sample_c1, sample_c2, sample_c3 = st.columns(3)
        sample_query = None
        if sample_c1.button("Main issues or blockers discussed?"):
            sample_query = "What were the main issues, blockers, or errors mentioned in this chat?"
        if sample_c2.button("Who initiated the key releases?"):
            sample_query = "Who were the primary coordinators or leads and what was shipped?"
        if sample_c3.button("Summarize peak activity day"):
            sample_query = "Summarize what happened on the peak activity day."

        user_query = st.text_input("Question:", value=sample_query or "", placeholder="e.g. Was the release deployed on time?")
        
        if st.button("Ask Assistant", type="secondary") or (sample_query and user_query):
            if user_query:
                with st.spinner("Querying chat transcript..."):
                    answer = ai_engine.ask_gemini_chat(user_query, df, api_key=user_api_key)
                    st.markdown("##### Answer:")
                    st.markdown(answer)

    # 3. Personas
    with ai_sub3:
        st.markdown("##### Participant Archetype Profiles")
        st.caption("Behavioral profiles inferred from messaging style, frequency, and tone.")
        
        if st.button("Generate Archetypes"):
            with st.spinner("Analyzing participant communication patterns..."):
                personas = ai_engine.generate_user_personas(df, api_key=user_api_key)
                st.session_state['personas'] = personas

        personas = st.session_state.get('personas', None)
        if personas is None:
            personas = ai_engine.generate_user_personas(df, api_key=user_api_key)

        p_cols = st.columns(min(len(personas), 4))
        for idx, p in enumerate(personas[:4]):
            with p_cols[idx]:
                badges_html = "".join([f"<span class='badge'>{b}</span>" for b in p['badges']])
                st.markdown(f"""
                <div class="persona-card">
                    <div style="font-size:0.75rem; opacity:0.6; text-transform:uppercase; font-weight:600;">Participant</div>
                    <div class="persona-user">{p['user']}</div>
                    <div class="persona-title">{p['title']}</div>
                    <div class="persona-desc">{p['description']}</div>
                    <div style="font-style:italic; color:#10b981; font-size:0.78rem; margin-top:6px;">"{p['catchphrase']}"</div>
                    <div style="margin-top:10px;">{badges_html}</div>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# Footer & Export
# -------------------------------------------------------------
st.markdown("---")
fcol1, fcol2 = st.columns([2, 1])
with fcol1:
    st.caption("ThreadModerator • Conversational Intelligence Engine built with Streamlit, Plotly, Scikit-Learn & Google Gemini.")
with fcol2:
    csv_buf = io.StringIO()
    if 'sentiment_score' in sentiment_res['df']:
        export_df = sentiment_res['df'][['date', 'user', 'message', 'sentiment_score', 'sentiment_label']].copy()
    else:
        export_df = df[['date', 'user', 'message']].copy()
    export_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Export Analyzed CSV",
        data=csv_buf.getvalue(),
        file_name="threadmoderator_analyzed.csv",
        mime="text/csv",
        use_container_width=True
    )