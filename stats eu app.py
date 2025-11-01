"""
EuroLeague Stats Comparator — Streamlit app (production-ready scaffolding)

This version chooses **Sportradar** as the primary production provider. If SPORTRADAR_KEY is not set or an error occurs, sample data is used.
"""
import os
import time
import json
import math
import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv, set_key, find_dotenv
from functools import wraps
from time import sleep
from pathlib import Path
from datetime import datetime

load_dotenv()

# ----------------- CONFIG -----------------
SPORTRADAR_KEY = os.getenv('SPORTRADAR_KEY')
SPORTRADAR_BASE = os.getenv('SPORTRADAR_BASE', 'https://api.sportradar.com')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '30'))
CACHE_DIR = Path('./cache')
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = int(os.getenv('CACHE_TTL', '60'))  # seconds

# ----------------- Utility: caching & retry -----------------

def cache_get(key):
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > CACHE_TTL:
            return None
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def cache_set(key, value):
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(value, f)
    except Exception:
        pass


def retry(max_attempts=4, backoff_factor=1.5):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = 1.0
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except requests.HTTPError as e:
                    status = getattr(e.response, 'status_code', None)
                    if status == 429:
                        # rate limited: backoff
                        sleep(delay)
                        delay *= backoff_factor
                        continue
                    # other HTTP errors: raise
                    raise
                except requests.RequestException:
                    sleep(delay)
                    delay *= backoff_factor
                    continue
            raise RuntimeError('Max retry attempts reached')
        return wrapper
    return deco

# ----------------- Sportradar Adapters -----------------

@retry(max_attempts=5)
def sportradar_get(path, params=None):
    if not SPORTRADAR_KEY:
        raise RuntimeError('SPORTRADAR_KEY not configured')
    url = SPORTRADAR_BASE.rstrip('/') + path
    params = params or {}
    params['api_key'] = SPORTRADAR_KEY
    r = requests.get(url, params=params, timeout=12)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 404:
        return None
    r.raise_for_status()


def fetch_sportradar_competitions():
    cache_key = 'sportradar_competitions'
    cached = cache_get(cache_key)
    if cached:
        return cached
    # Global basketball competitions list (example endpoint)
    data = sportradar_get('/basketball/trial/v2/en/competitions.json')
    # Note: endpoint path/version may vary; adjust per your account's version access
    if not data:
        return []
    comps = [{'id': c.get('id'), 'name': c.get('name')} for c in data.get('competitions', [])]
    cache_set(cache_key, comps)
    return comps


def fetch_sportradar_players_by_competition(comp_id):
    cache_key = f'sr_players_{comp_id}'
    cached = cache_get(cache_key)
    if cached:
        return pd.DataFrame(cached)
    # Example: competitor roster feed
    data = sportradar_get(f'/basketball/trial/v2/en/competitions/{comp_id}/players.json')
    rows = []
    if not data:
        return pd.DataFrame(rows)
    for p in data.get('players', []):
        rows.append({
            'player_id': p.get('id'),
            'name': (p.get('firstname', '') + ' ' + p.get('lastname', '')).strip(),
            'team': p.get('current_team', {}).get('name', '') if p.get('current_team') else '',
            'pts': p.get('pts', 0),
            'two_points_made': p.get('two_points_made', 0),
            'three_points_made': p.get('three_points_made', 0),
            'reb': p.get('reb', 0),
            'ast': p.get('ast', 0),
            'stl': p.get('stl', 0),
            'blk': p.get('blk', 0),
            'min': p.get('min', 0),
            'fgm': p.get('fgm', 0),
            'fga': p.get('fga', 0),
            'fta': p.get('fta', 0),
            'ftm': p.get('ftm', 0),
            'turnovers': p.get('turnovers', 0)
        })
    df = pd.DataFrame(rows)
    cache_set(cache_key, df.to_dict('records'))
    return df


def fetch_sportradar_game_boxscore(game_id):
    cache_key = f'sr_box_{game_id}'
    cached = cache_get(cache_key)
    if cached:
        return cached
    data = sportradar_get(f'/basketball/trial/v2/en/games/{game_id}/boxscore.json')
    if not data:
        return None
    cache_set(cache_key, data)
    return data


def fetch_sportradar_game_pbp(game_id):
    cache_key = f'sr_pbp_{game_id}'
    cached = cache_get(cache_key)
    if cached:
        return cached
    data = sportradar_get(f'/basketball/trial/v2/en/games/{game_id}/pbp.json')
    if not data:
        return None
    cache_set(cache_key, data)
    return data

# ----------------- Analytics helpers -----------------

COMPARABLE_STATS = [
    ('pts', 'Points'),
    ('two_points_made', '2P Made'),
    ('three_points_made', '3P Made'),
    ('reb', 'Rebounds'),
    ('ast', 'Assists'),
    ('stl', 'Steals'),
    ('blk', 'Blocks'),
    ('min', 'Minutes'),
    ('fga', 'FGA'),
    ('fgm', 'FGM'),
    ('fta', 'FTA'),
    ('ftm', 'FTM'),
    ('turnovers', 'Turnovers'),
]


def per36(stat_value, minutes_played):
    try:
        minutes_played = float(minutes_played)
        if minutes_played <= 0:
            return 0.0
        return stat_value * (36.0 / minutes_played)
    except Exception:
        return 0.0


def efg_percentage(fgm, fga, three_pm):
    try:
        fga = float(fga)
        if fga == 0:
            return 0.0
        return (float(fgm) + 0.5 * float(three_pm)) / fga
    except Exception:
        return 0.0


def ts_percentage(pts, fga, fta):
    try:
        denom = 2 * (float(fga) + 0.44 * float(fta))
        if denom == 0:
            return 0.0
        return float(pts) / denom
    except Exception:
        return 0.0


def efficiency_eff(row):
    try:
        pts = float(row.get('pts', 0))
        reb = float(row.get('reb', 0))
        ast = float(row.get('ast', 0))
        stl = float(row.get('stl', 0))
        blk = float(row.get('blk', 0))
        fga = float(row.get('fga', 0))
        fgm = float(row.get('fgm', 0))
        fta = float(row.get('fta', 0))
        ftm = float(row.get('ftm', 0))
        tov = float(row.get('turnovers', 0))
        eff = pts + reb + ast + stl + blk - (fga - fgm) - (fta - ftm) - tov
        return eff
    except Exception:
        return 0.0


def estimate_possessions(team_stats):
    try:
        fga = float(team_stats.get('fga', 0))
        fta = float(team_stats.get('fta', 0))
        orbd = float(team_stats.get('orb', team_stats.get('offensive_rebounds', 0) or 0))
        tov = float(team_stats.get('turnovers', team_stats.get('turnover', 0) or 0))
        if orbd == 0:
            trb = float(team_stats.get('reb', team_stats.get('total_rebounds', 0) or 0))
            orbd = 0.2 * trb
        poss = fga + 0.44 * fta - orbd + tov
        return max(poss, 1)
    except Exception:
        return 1

# ----------------- Visualization -----------------

def plot_bar_comparison(stats_keys, p1_stats, p2_stats, labels):
    fig = go.Figure()
    x = [disp for _, disp in stats_keys]
    fig.add_trace(go.Bar(name=labels[0], x=x, y=[p1_stats.get(k, 0) for k, _ in stats_keys]))
    fig.add_trace(go.Bar(name=labels[1], x=x, y=[p2_stats.get(k, 0) for k, _ in stats_keys]))
    fig.update_layout(barmode='group', title='Player comparison — bar')
    return fig


def plot_radar_comparison(stats_keys, p1_stats, p2_stats, labels):
    categories = [disp for _, disp in stats_keys]
    p1 = [p1_stats.get(k, 0) for k, _ in stats_keys]
    p2 = [p2_stats.get(k, 0) for k, _ in stats_keys]
    categories += [categories[0]]
    p1 += [p1[0]]
    p2 += [p2[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=p1, theta=categories, fill='toself', name=labels[0]))
    fig.add_trace(go.Scatterpolar(r=p2, theta=categories, fill='toself', name=labels[1]))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, title='Player comparison — radar')
    return fig

# ----------------- Sample fallback data -----------------

def sample_data_players():
    players = [
        {'player_id': 1, 'name': 'Nikola M.', 'team': 'Anadolu Efes', 'pts': 18.6, 'two_points_made': 6.1, 'three_points_made': 2.0, 'reb': 5.4, 'ast': 3.1, 'stl':1.2, 'blk':0.6, 'min':28.3, 'fgm':7.2, 'fga':14.8, 'fta':3.2, 'ftm':2.2, 'turnovers':2.1},
        {'player_id': 2, 'name': 'Luka J.', 'team': 'Real Madrid', 'pts': 15.2, 'two_points_made':5.0, 'three_points_made':2.6, 'reb':3.2, 'ast':5.7, 'stl':0.9, 'blk':0.2, 'min':26.9, 'fgm':6.1, 'fga':13.5, 'fta':2.1, 'ftm':1.6, 'turnovers':1.9},
        {'player_id': 3, 'name': 'Sergio R.', 'team': 'FC Barcelona', 'pts': 9.4, 'two_points_made':3.6, 'three_points_made':1.1, 'reb':4.8, 'ast':1.9, 'stl':0.7, 'blk':0.3, 'min':18.7, 'fgm':4.1, 'fga':9.2, 'fta':1.0, 'ftm':0.8, 'turnovers':0.8},
    ]
    return pd.DataFrame(players)

# ----------------- Streamlit app -----------------

def main():
    # small cosmetic: handwritten-looking font + gentle card styling to feel "handmade"
    st.set_page_config(page_title='EuroLeague Stats Comparator (Handmade)', layout='wide')
    st.markdown(
        """<style>
        @import url('https://fonts.googleapis.com/css2?family=Indie+Flower&display=swap');
        .stApp h1 {font-family: 'Indie Flower', cursive; font-size:40px; color:#2b2b2b; margin-bottom:0;}
        .header-note {font-family: 'Indie Flower', cursive; color:#6b6b6b; margin-top:4px;}
        .card {background:#fffef6; padding:12px; border-radius:8px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); margin-bottom:10px;}
        .author {font-size:12px; color:#666; margin-top:6px;}
        .small-muted {font-size:12px; color:#777;}
        </style>""",
        unsafe_allow_html=True
    )

    st.title("EuroLeague — Player & Team Stats Comparator")
    st.markdown('<div class="header-note">A small, hand-crafted stats viewer — quick comparisons and a friendly UI ✨</div>', unsafe_allow_html=True)

    # Sidebar personal touch
    st.sidebar.header("Settings & Notes")
    st.sidebar.markdown('<div class="card">Made by a student dev — proof-of-concept. If something looks off, it was probably written late at night 🙂</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="small-muted">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="author">Tip: pick two players and try the radar view — I like that one.</div>', unsafe_allow_html=True)

    st.sidebar.header('Data & Settings')
    # Allow entering a Sportradar API key at runtime (and optionally save to .env)
    runtime_key = st.sidebar.text_input("Sportradar API key (optional)", value="", type="password", help="Enter your Sportradar API key to enable live data")
    if runtime_key:
        # override runtime variable and environment for this run
        global SPORTRADAR_KEY
        SPORTRADAR_KEY = runtime_key
        os.environ['SPORTRADAR_KEY'] = runtime_key
        if st.sidebar.button("Save key to .env"):
            env_file = find_dotenv(usecwd=True)
            if not env_file:
                # create .env in project root
                env_file = str(Path.cwd() / ".env")
            try:
                set_key(env_file, "SPORTRADAR_KEY", runtime_key)
                st.sidebar.success("SPORTRADAR_KEY saved to .env")
            except Exception as e:
                st.sidebar.error(f"Failed to save .env: {e}")

    backend = st.sidebar.selectbox('Choose data backend', ['Sportradar (recommended)', 'Sample (offline)', 'API-Basketball (RapidAPI)'])
    season = st.sidebar.text_input('Season code (e.g. 2024)', value='2024')
    realtime = st.sidebar.checkbox('Enable realtime polling (I can keep an eye on live games for you)', value=False)
    poll_interval = st.sidebar.number_input('Polling interval (seconds)', min_value=5, max_value=300, value=POLL_INTERVAL)

    # Fetch players based on backend selection
    if backend == 'Sportradar (recommended)':
        if not SPORTRADAR_KEY:
            st.warning("SPORTRADAR_KEY not set. Falling back to sample data.")
            players_df = sample_data_players()
        else:
            try:
                comps = fetch_sportradar_competitions()
                euro_comp = None
                for c in comps:
                    if 'euro' in (c.get('name','').lower()):
                        euro_comp = c
                        break
                if euro_comp:
                    players_df = fetch_sportradar_players_by_competition(euro_comp['id'])
                    if players_df.empty:
                        st.warning("No players found in Sportradar data. Using sample data.")
                        players_df = sample_data_players()
                else:
                    st.info("No Euroleague-like competition found. Using sample data.")
                    players_df = sample_data_players()
            except Exception as e:
                st.error("Error fetching Sportradar data: " + str(e))
                players_df = sample_data_players()
    elif backend == 'API-Basketball (RapidAPI)':
        players_df = sample_data_players()
    else:
        players_df = sample_data_players()

    players_df['display'] = players_df['name'] + ' — ' + players_df['team'].fillna('')

    menu = st.sidebar.radio('Page', ['Player vs Player', 'Team Comparison', 'Play-by-Play / Live'])

    if menu == 'Player vs Player':
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader('Pick players')
            st.markdown("Choose two players to compare — be bold, try improbable matchups.")
            p1_display = st.selectbox('Player 1', options=players_df['display'].tolist())
            p2_display = st.selectbox('Player 2', options=[d for d in players_df['display'].tolist() if d != p1_display])
            stats_to_use = st.multiselect('Select stats to compare', [disp for _, disp in COMPARABLE_STATS], default=[disp for _, disp in COMPARABLE_STATS][:8])
            per36_toggle = st.checkbox('Show per-36 normalized values (handy for minutes differences)', value=False)
            adv_toggle = st.checkbox('Include advanced metrics (eFG%, TS%, EFF)', value=True)
        with col2:
            p1_row = players_df[players_df['display'] == p1_display].iloc[0].to_dict()
            p2_row = players_df[players_df['display'] == p2_display].iloc[0].to_dict()
            def build_stats(row):
                s = {k: row.get(k, 0) for k, _ in COMPARABLE_STATS}
                s['efg'] = efg_percentage(row.get('fgm', 0), row.get('fga', 0), row.get('three_points_made', 0))
                s['ts'] = ts_percentage(row.get('pts', 0), row.get('fga', 0), row.get('fta', 0))
                s['eff'] = efficiency_eff(row)
                return s
            p1_stats = build_stats(p1_row)
            p2_stats = build_stats(p2_row)
            if per36_toggle:
                for k, _ in COMPARABLE_STATS:
                    p1_stats[k] = per36(p1_stats.get(k, 0), p1_row.get('min', 0))
                    p2_stats[k] = per36(p2_stats.get(k, 0), p2_row.get('min', 0))
                p1_stats['eff'] = per36(p1_stats['eff'], p1_row.get('min', 0))
                p2_stats['eff'] = per36(p2_stats['eff'], p2_row.get('min', 0))
            selected_pairs = [(k, disp) for k, disp in COMPARABLE_STATS if disp in stats_to_use]
            if not selected_pairs:
                selected_pairs = COMPARABLE_STATS[:8]
            table = pd.DataFrame({
                p1_row['name']: [p1_stats.get(k, 0) for k, _ in selected_pairs],
                p2_row['name']: [p2_stats.get(k, 0) for k, _ in selected_pairs]
            }, index=[disp for _, disp in selected_pairs])
            st.dataframe(table)
            st.plotly_chart(plot_bar_comparison(selected_pairs, p1_stats, p2_stats, (p1_row['name'], p2_row['name'])), use_container_width=True)
            st.plotly_chart(plot_radar_comparison(selected_pairs, p1_stats, p2_stats, (p1_row['name'], p2_row['name'])), use_container_width=True)
            if adv_toggle:
                adv_table = pd.DataFrame({
                    p1_row['name']: [p1_stats['efg'], p1_stats['ts'], p1_stats['eff']],
                    p2_row['name']: [p2_stats['efg'], p2_stats['ts'], p2_stats['eff']]
                }, index=['eFG%', 'TS%', 'EFF'])
                st.dataframe(adv_table)
    elif menu == 'Team Comparison':
        st.subheader('Team vs Team season comparison')
        teams = sorted(players_df['team'].dropna().unique().tolist())
        t1 = st.selectbox('Team 1', teams)
        t2 = st.selectbox('Team 2', [t for t in teams if t != t1] or teams)
        def aggregate_team(team_name):
            df = players_df[players_df['team'] == team_name]
            agg = {}
            for k, _ in COMPARABLE_STATS:
                agg[k] = df[k].sum() if k in df.columns else 0
            agg['efg'] = efg_percentage(
                df['fgm'].sum() if 'fgm' in df.columns else 0,
                df['fga'].sum() if 'fga' in df.columns else 0,
                df['three_points_made'].sum() if 'three_points_made' in df.columns else 0)
            agg['ts'] = ts_percentage(
                df['pts'].sum() if 'pts' in df.columns else 0,
                
                df['fga'].sum() if 'fga' in df.columns else 0,
                df['fta'].sum() if 'fta' in df.columns else 0)
            agg['eff'] = df.apply(efficiency_eff, axis=1).sum() if not df.empty else 0
            return agg
        t1_stats = aggregate_team(t1)
        t2_stats = aggregate_team(t2)
        selected_pairs = COMPARABLE_STATS[:8]
        st.dataframe(pd.DataFrame({
            t1: [t1_stats[k] for k, _ in selected_pairs],
            t2: [t2_stats[k] for k, _ in selected_pairs]
        }, index=[disp for _, disp in selected_pairs]))
        st.plotly_chart(plot_bar_comparison(selected_pairs, t1_stats, t2_stats, (t1, t2)), use_container_width=True)
    else:  # Play-by-Play / Live
        st.subheader('Play-by-play / Live game feed (Sportradar)')
        if backend != 'Sportradar (recommended)':
            st.info('Switch to Sportradar backend and set SPORTRADAR_KEY to enable live data')
        else:
            try:
                games_cache_key = 'sr_recent_games'
                games = cache_get(games_cache_key)
                if not games:
                    comps = fetch_sportradar_competitions()
                    games = []
                    for comp in comps[:5]:
                        feed = sportradar_get(f"/basketball/trial/v2/en/competitions/{comp['id']}/schedule.json")
                        if feed and 'games' in feed:
                            for g in feed['games']:
                                games.append({
                                    'id': g.get('id'),
                                    'home': g.get('home', {}).get('name'),
                                    'away': g.get('away', {}).get('name'),
                                    'scheduled': g.get('scheduled')
                                })
                    cache_set(games_cache_key, games)
                game_options = [f"{g.get('home')} vs {g.get('away')} — {g.get('scheduled')} ({g.get('id')})" for g in (games or [])]
                if game_options:
                    sel = st.selectbox('Pick game', options=game_options)
                    game_id = sel.split('(')[-1].rstrip(')')
                    pbp = fetch_sportradar_game_pbp(game_id)
                    box = fetch_sportradar_game_boxscore(game_id)
                    st.write('Play-by-play:')
                    if pbp and 'events' in pbp:
                        for ev in pbp.get('events', [])[:200]:
                            st.markdown(f"- {ev.get('clock','')} — {ev.get('description','')}")
                    else:
                        st.info('No play-by-play available in this demo or the feed is empty.')
                else:
                    st.info('No recent games available.')
            except Exception as e:
                st.error("Error fetching live game data: " + str(e))
    # Deployment templates
    if st.sidebar.checkbox('Show deployment templates'):
        dockerfile = '''FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir streamlit requests pandas plotly python-dotenv
EXPOSE 8501
CMD ["streamlit", "run", "euroleague_stats_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]'''
        proc = 'web: streamlit run euroleague_stats_streamlit.py --server.port=$PORT'
        st.code(dockerfile, language='dockerfile')
        st.code(proc, language='bash')
    if realtime:
        st.sidebar.info(f'Realtime enabled — polling every {int(poll_interval)} seconds.')
        time.sleep(int(poll_interval))
        # Cross-version Streamlit: prefer st.rerun(), fallback to experimental_rerun(), otherwise stop execution.
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            # If neither rerun API exists, stop this run (user can refresh manually)
            st.stop()

if __name__ == '__main__':
    main()
