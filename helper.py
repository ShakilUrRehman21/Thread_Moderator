import re
import os
import pandas as pd
import numpy as np
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import emoji
import plotly.express as px
import plotly.graph_objects as go
from ai_engine import get_stopwords

extract = URLExtract()

# Unified Color Palette for Clean Modern Dark Theme
PALETTE = {
    'bg': '#0f172a',
    'card': '#1e293b',
    'border': '#334155',
    'primary': '#10b981',       # Clean Emerald
    'primary_light': '#34d399',
    'secondary': '#38bdf8',     # Sky Blue
    'accent': '#6366f1',        # Indigo
    'warning': '#f59e0b',       # Amber
    'danger': '#f43f5e',        # Rose
    'text': '#f8fafc',
    'text_muted': '#94a3b8'
}

PLOTLY_TEMPLATE = None


def fetch_stats(selected_user: str, df: pd.DataFrame) -> dict:
    """
    Computes high-level statistics for the selected user or overall group.
    """
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    valid_msgs = data[(~data['is_system'])]
    num_messages = len(valid_msgs)

    # Word count
    words = []
    for msg in valid_msgs['message']:
        words.extend(str(msg).split())
    total_words = len(words)

    # Media count
    num_media = int(valid_msgs['is_media'].sum())

    # Links count
    links = []
    for msg in valid_msgs['message']:
        links.extend(extract.find_urls(str(msg)))
    total_links = len(links)

    # Active participants & time span
    active_participants = data['user'].nunique()
    if 'group_notification' in data['user'].values:
        active_participants -= 1
    active_participants = max(1, active_participants)

    days_span = max(1, (data['date'].max() - data['date'].min()).days + 1) if not data.empty else 1
    avg_per_day = round(num_messages / days_span, 1)

    return {
        'num_messages': num_messages,
        'total_words': total_words,
        'num_media': num_media,
        'total_links': total_links,
        'active_participants': active_participants,
        'days_span': days_span,
        'avg_per_day': avg_per_day
    }


def most_busy_users(df: pd.DataFrame, top_n: int = 10):
    """Returns top active users by message volume and percentage."""
    valid = df[(~df['is_system']) & (df['user'] != 'group_notification')]
    counts = valid['user'].value_counts().head(top_n)
    
    total = len(valid)
    pct_df = (round((counts / total) * 100, 2)).reset_index()
    pct_df.columns = ['User', 'Percentage (%)']
    pct_df['Message Count'] = counts.values
    return counts, pct_df


def create_wordcloud(selected_user: str, df: pd.DataFrame):
    """Generates a high-definition WordCloud object with dark theme palette."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    valid = data[(~data['is_system']) & (~data['is_media']) & (~data['is_deleted'])]
    stops = get_stopwords()

    clean_words = []
    for msg in valid['message']:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', str(msg).lower())
        clean_words.extend([w for w in tokens if w not in stops])

    if not clean_words:
        clean_words = ['chat', 'conversation', 'analysis', 'data']

    text_corpus = " ".join(clean_words)
    wc = WordCloud(
        width=800,
        height=450,
        min_font_size=10,
        background_color=None,
        mode="RGBA",
        colormap='viridis',
        max_words=120,
        contour_width=0
    )
    return wc.generate(text_corpus)


def most_common_words(selected_user: str, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Returns DataFrame of top most common words with stopwords removed."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    valid = data[(~data['is_system']) & (~data['is_media']) & (~data['is_deleted'])]
    stops = get_stopwords()

    words = []
    for msg in valid['message']:
        tokens = re.findall(r'\b[a-zA-Z]{3,}\b', str(msg).lower())
        words.extend([w for w in tokens if w not in stops])

    common = Counter(words).most_common(top_n)
    if not common:
        return pd.DataFrame(columns=['Word', 'Frequency'])
    
    df_common = pd.DataFrame(common, columns=['Word', 'Frequency'])
    return df_common


def emoji_helper(selected_user: str, df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extracts top emojis with count and percentage."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    emojis = []
    for msg in data['message']:
        emojis.extend([c for c in str(msg) if c in emoji.EMOJI_DATA])

    if not emojis:
        return pd.DataFrame(columns=['Emoji', 'Count', 'Percentage (%)'])

    counts = pd.Series(emojis).value_counts().head(top_n).reset_index()
    counts.columns = ['Emoji', 'Count']
    total = len(emojis)
    counts['Percentage (%)'] = (counts['Count'] / total * 100).round(1)
    
    # Add emoji description name
    descriptions = []
    for e in counts['Emoji']:
        desc = emoji.demojize(e).replace(':', '').replace('_', ' ').title()
        descriptions.append(desc)
    counts['Description'] = descriptions
    return counts


def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates messages by Year-Month."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    timeline = data.groupby(['year', 'month_num', 'month'])['message'].count().reset_index()
    timeline['time'] = timeline['month'] + " " + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates messages by single date."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    daily = data.groupby('only_date')['message'].count().reset_index()
    daily['date'] = pd.to_datetime(daily['only_date'])
    daily.sort_values('date', inplace=True)
    daily['rolling_avg'] = daily['message'].rolling(window=7, min_periods=1).mean().round(1)
    daily['cumulative'] = daily['message'].cumsum()
    return daily


def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    return data['day_name'].value_counts().reindex(order).fillna(0)


def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    return data['month'].value_counts().reindex(months).dropna()


def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    """Builds pivot table for Day of Week vs Period."""
    data = df.copy()
    if selected_user != 'Overall':
        data = data[data['user'] == selected_user]

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap = data.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
    heatmap = heatmap.reindex(days_order).fillna(0)
    return heatmap


# -------------------------------------------------------------
# Interactive Plotly Chart Builders (Ultra-Modern Dark UI)
# -------------------------------------------------------------

def plot_interactive_timeline(daily_df: pd.DataFrame) -> go.Figure:
    """Interactive line & volume timeline with range slider."""
    fig = go.Figure()
    
    # Bar for daily count
    fig.add_trace(go.Bar(
        x=daily_df['date'],
        y=daily_df['message'],
        name='Daily Messages',
        marker_color='rgba(16, 185, 129, 0.45)',
        hovertemplate='%{x|%d %b %Y}<br>Messages: %{y}<extra></extra>'
    ))

    # Smoothed 7-day rolling average
    fig.add_trace(go.Scatter(
        x=daily_df['date'],
        y=daily_df['rolling_avg'],
        name='7-Day Moving Avg',
        mode='lines',
        line=dict(color=PALETTE['secondary'], width=3, shape='spline'),
        hovertemplate='%{x|%d %b %Y}<br>7-Day Avg: %{y:.1f}<extra></extra>'
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Message Velocity & Activity Over Time',
        xaxis_title='Date',
        yaxis_title='Messages',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    return fig


def plot_sentiment_gauge(sentiment_summary: dict) -> go.Figure:
    """Donut chart showing sentiment proportions."""
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        sentiment_summary.get('Positive', 0),
        sentiment_summary.get('Neutral', 0),
        sentiment_summary.get('Negative', 0)
    ]
    colors = [PALETTE['primary'], '#64748b', PALETTE['danger']]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='#0f172a', width=2)),
        textinfo='label+percent',
        hoverinfo='label+value+percent'
    )])

    score = sentiment_summary.get('avg_score', 0.0)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Sentiment Distribution',
        annotations=[dict(
            text=f"Score<br><b>{score:+.2f}</b>",
            x=0.5, y=0.5,
            font_size=17,
            showarrow=False,
            font_color=PALETTE['primary_light'] if score >= 0 else PALETTE['danger']
        )],
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_sentiment_trajectory(daily_sent_df: pd.DataFrame) -> go.Figure:
    """Smoothed sentiment trend line over time."""
    if daily_sent_df.empty or 'date' not in daily_sent_df:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_sent_df['date'],
        y=daily_sent_df['rolling_sentiment'],
        mode='lines',
        name='Sentiment (7-Day Avg)',
        line=dict(color=PALETTE['primary_light'], width=2.5, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.12)'
    ))

    # Neutral reference line at 0.0
    fig.add_hline(y=0.0, line_dash="dash", line_color="#475569", annotation_text="Baseline (0.0)")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Sentiment Trajectory (Group Mood Over Time)',
        xaxis_title='Date',
        yaxis_title='Compound Polarity (-1 to +1)',
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_intent_radar(intent_counts: dict) -> go.Figure:
    """Radar chart displaying conversational intents."""
    if not intent_counts:
        return go.Figure()

    categories = list(intent_counts.keys())
    values = list(intent_counts.values())
    # Close polygon
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.25)',
        line=dict(color=PALETTE['accent'], width=2),
        name='Message Intents'
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Conversational Intent Breakdown',
        polar=dict(
            radialaxis=dict(visible=True, showline=False, color='#64748b'),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=30, r=30, t=40, b=30),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_heatmap_interactive(heatmap_df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap for Day of Week vs Hour Period."""
    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Hour Period", y="Day of Week", color="Messages"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        color_continuous_scale="Tealgrn",
        aspect="auto"
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Weekly Activity Heatmap',
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_busy_users_interactive(busy_users_series: pd.Series) -> go.Figure:
    """Horizontal bar chart for top contributors."""
    df_plot = busy_users_series.reset_index()
    df_plot.columns = ['User', 'Messages']
    df_plot = df_plot.sort_values('Messages', ascending=True)

    fig = go.Figure(go.Bar(
        x=df_plot['Messages'],
        y=df_plot['User'],
        orientation='h',
        marker=dict(
            color=df_plot['Messages'],
            colorscale='Blues',
            line=dict(color='#0f172a', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Messages: %{x}<extra></extra>'
    ))

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='Most Active Contributors',
        xaxis_title='Messages',
        yaxis_title='',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_network_interactive(network_data: dict) -> go.Figure:
    """Interactive 2D Social Interaction Network Graph in Plotly."""
    nodes = network_data.get('nodes', [])
    edges = network_data.get('edges', [])

    if not nodes:
        return go.Figure()

    # Draw edge lines
    edge_x = []
    edge_y = []
    for e in edges:
        edge_x.extend([e['x0'], e['x1'], None])
        edge_y.extend([e['y0'], e['y1'], None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(100, 116, 139, 0.35)'),
        hoverinfo='none',
        mode='lines'
    )

    # Draw nodes
    node_x = [n['x'] for n in nodes]
    node_y = [n['y'] for n in nodes]
    node_text = [
        f"<b>{n['name']}</b><br>Total Connections: {n['total_degree']}<br>Replies Received: {n['in_degree']}<br>Replies Sent: {n['out_degree']}"
        for n in nodes
    ]
    node_size = [max(12, min(40, 10 + n['total_degree'] * 1.5)) for n in nodes]
    node_color = [n['total_degree'] for n in nodes]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n['name'][:12] for n in nodes],
        textposition="bottom center",
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Blues',
            reversescale=True,
            color=node_color,
            size=node_size,
            colorbar=dict(
                thickness=12,
                title="Connections",
                xanchor="left"
            ),
            line=dict(width=2, color='#ffffff')
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
        layout=go.Layout(
            template=PLOTLY_TEMPLATE,
            title='🕸️ Conversational Interaction Network (Who Replies to Whom)',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=20, r=20, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )
    return fig


def plot_word_frequency_interactive(word_df: pd.DataFrame) -> go.Figure:
    """Bar chart of top words."""
    if word_df.empty:
        return go.Figure()
    
    top = word_df.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top['Frequency'],
        y=top['Word'],
        orientation='h',
        marker=dict(color=PALETTE['secondary'], line=dict(color='#0e1117', width=1)),
        hovertemplate='<b>%{y}</b>: %{x} times<extra></extra>'
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='💬 Most Frequent Meaningful Words',
        xaxis_title='Frequency',
        yaxis_title='',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def plot_emoji_distribution(emoji_df: pd.DataFrame) -> go.Figure:
    """Pie/Donut chart for top emojis."""
    if emoji_df.empty:
        return go.Figure()

    top_emojis = emoji_df.head(8)
    fig = go.Figure(data=[go.Pie(
        labels=[f"{row['Emoji']} ({row['Description'][:14]})" for _, row in top_emojis.iterrows()],
        values=top_emojis['Count'],
        hole=0.45,
        textinfo='label+percent',
        hoverinfo='label+value'
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title='😀 Top Emoji Frequency',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig
