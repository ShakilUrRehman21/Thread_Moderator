import re
import pandas as pd
from datetime import datetime


def preprocess(data: str) -> pd.DataFrame:
    """
    Robust WhatsApp Chat Preprocessor
    Supports:
    - 24-hour and 12-hour (AM/PM) formats
    - Android and iOS formats (with or without brackets)
    - Unicode non-breaking space (\\u202f, \\xa0) variations
    - Multi-line messages
    - Automatic date parsing with dayfirst=True
    - System notification filtering and classification
    """
    if not data or not data.strip():
        return pd.DataFrame()

    # Normalize unicode spaces and control characters
    data = data.replace('\u202f', ' ').replace('\xa0', ' ').replace('\r\n', '\n').replace('\r', '\n')
    # Remove left-to-right / right-to-left marks
    data = re.sub(r'[\u200e\u200f\u202a-\u202e]', '', data)

    # Universal WhatsApp timestamp pattern:
    # Matches:
    # 14/05/24, 20:45 - 
    # 14/05/2024, 20:45 - 
    # 14/05/24, 8:45 pm - 
    # 14/05/2024, 8:45 PM - 
    # [14/05/24, 8:45:15 PM] 
    # [14/05/2024, 20:45:15] 
    # 14.05.24, 20:45 -
    pattern = r'(?:^|\n)(\[?\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?(?:[\s]*(?:AM|PM|am|pm|a\.m\.|p\.m\.))?\]?(?:\s*-\s*|\s+))'

    parts = re.split(pattern, data)

    # First element before the first timestamp is usually empty
    if parts and parts[0].strip() == '':
        parts = parts[1:]

    raw_dates = parts[0::2]
    raw_messages = parts[1::2]

    if not raw_dates or not raw_messages:
        # Fallback if split didn't catch properly: try line-by-line
        return _fallback_line_preprocess(data)

    # Clean raw date strings
    cleaned_dates = []
    for d in raw_dates:
        cleaned = d.strip(' []-\n\t')
        cleaned_dates.append(cleaned)

    # Parse datetimes safely
    parsed_dates = pd.to_datetime(cleaned_dates, format='mixed', dayfirst=True, errors='coerce')

    users = []
    messages = []
    is_system_list = []

    # System message keywords
    system_indicators = [
        "messages and calls are end-to-end encrypted",
        "joined using this group's invite link",
        "left",
        "added",
        "removed",
        "changed the group description",
        "changed the subject to",
        "changed this group's icon",
        "created group",
        "security code changed",
        "you're now an admin",
        "you are now an admin"
    ]

    for raw_msg in raw_messages:
        msg = raw_msg.strip()
        # Look for "Sender Name: Message content"
        entry = re.split(r'([\w\W]+?):\s', msg, maxsplit=1)
        if len(entry) >= 3:
            user = entry[1].strip()
            # Clean leading ~ (WhatsApp pushname marker)
            if user.startswith('~'):
                user = user.lstrip('~ ').strip()
            users.append(user)
            messages.append(entry[2].strip())
            is_system_list.append(False)
        else:
            # System notification or group event
            clean_text = entry[0].strip()
            users.append('group_notification')
            messages.append(clean_text)
            is_system_list.append(True)

    df = pd.DataFrame({
        'date': parsed_dates,
        'user': users,
        'message': messages,
        'is_system': is_system_list
    })

    # Drop any records with unparseable dates
    df.dropna(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Feature Engineering
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    # Period intervals (e.g. 14-15)
    periods = []
    for h in df['hour']:
        if h == 23:
            periods.append("23-00")
        elif h == 0:
            periods.append("00-01")
        else:
            periods.append(f"{h:02d}-{(h + 1):02d}")
    df['period'] = periods

    # Time of Day category
    def categorize_time_of_day(hour):
        if 0 <= hour < 6:
            return "Late Night (00-06)"
        elif 6 <= hour < 12:
            return "Morning (06-12)"
        elif 12 <= hour < 17:
            return "Afternoon (12-17)"
        elif 17 <= hour < 21:
            return "Evening (17-21)"
        else:
            return "Night (21-24)"

    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)

    # Message attributes
    df['char_count'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['is_media'] = df['message'].apply(lambda x: '<Media omitted>' in x or '<image omitted>' in x or '<video omitted>' in x or '<audio omitted>' in x)
    df['is_deleted'] = df['message'].apply(lambda x: 'this message was deleted' in x.lower() or 'you deleted this message' in x.lower())

    return df


def _fallback_line_preprocess(data: str) -> pd.DataFrame:
    """Fallback line-by-line processor for non-standard formats."""
    lines = data.split('\n')
    parsed_rows = []
    curr_date = None
    curr_user = None
    curr_msg = []

    pattern = re.compile(r'(\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4})[,\s]+(\d{1,2}:\d{2}(?::\d{2})?(?:[\s]*(?:AM|PM|am|pm))?)')

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        
        match = pattern.search(line_clean)
        if match:
            # Save previous message
            if curr_date and curr_msg:
                full_msg = "\n".join(curr_msg)
                entry = re.split(r'([\w\W]+?):\s', full_msg, maxsplit=1)
                user = entry[1].strip() if len(entry) >= 3 else 'group_notification'
                text = entry[2].strip() if len(entry) >= 3 else entry[0].strip()
                parsed_rows.append((curr_date, user, text))
            
            # Start new message
            date_str = match.group(0)
            curr_date = pd.to_datetime(date_str, format='mixed', dayfirst=True, errors='coerce')
            rest = line_clean[match.end():].lstrip(' -:')
            curr_msg = [rest]
        else:
            if curr_msg:
                curr_msg.append(line_clean)

    if curr_date and curr_msg:
        full_msg = "\n".join(curr_msg)
        entry = re.split(r'([\w\W]+?):\s', full_msg, maxsplit=1)
        user = entry[1].strip() if len(entry) >= 3 else 'group_notification'
        text = entry[2].strip() if len(entry) >= 3 else entry[0].strip()
        parsed_rows.append((curr_date, user, text))

    if not parsed_rows:
        return pd.DataFrame()

    df = pd.DataFrame(parsed_rows, columns=['date', 'user', 'message'])
    df['is_system'] = df['user'] == 'group_notification'
    df.dropna(subset=['date'], inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['is_weekend'] = df['date'].dt.dayofweek >= 5

    periods = []
    for h in df['hour']:
        if h == 23:
            periods.append("23-00")
        elif h == 0:
            periods.append("00-01")
        else:
            periods.append(f"{h:02d}-{(h + 1):02d}")
    df['period'] = periods

    def categorize_time_of_day(hour):
        if 0 <= hour < 6:
            return "Late Night (00-06)"
        elif 6 <= hour < 12:
            return "Morning (06-12)"
        elif 12 <= hour < 17:
            return "Afternoon (12-17)"
        elif 17 <= hour < 21:
            return "Evening (17-21)"
        else:
            return "Night (21-24)"

    df['time_of_day'] = df['hour'].apply(categorize_time_of_day)
    df['char_count'] = df['message'].apply(len)
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    df['is_media'] = df['message'].apply(lambda x: '<Media omitted>' in x)
    df['is_deleted'] = df['message'].apply(lambda x: 'this message was deleted' in x.lower() or 'you deleted this message' in x.lower())

    return df
