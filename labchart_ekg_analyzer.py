import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from io import StringIO, BytesIO
import zipfile
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("LabChart EKG Analyzer")

# Split into injection sections
@st.cache_data(show_spinner=False)
def split_sections(file_bytes):
    lines = file_bytes.decode('ISO-8859-1').splitlines()
    channel_line = next(line for line in lines if line.startswith("ChannelTitle="))
    channel_titles = channel_line.replace("ChannelTitle=", "").strip().split('\t')
    columns = ['Time (s)'] + channel_titles

    data_starts = [i for i, line in enumerate(lines) if line.strip() and (line[0].isdigit() or line[0] == '.')]
    sections = []
    current_start = data_starts[0]
    for i in range(1, len(data_starts)):
        try:
            prev = float(lines[data_starts[i - 1]].split('\t')[0])
            this = float(lines[data_starts[i]].split('\t')[0])
            if this < prev:
                sections.append((current_start, data_starts[i - 1] + 1))
                current_start = data_starts[i]
        except:
            continue
    sections.append((current_start, len(lines)))
    return lines, columns, sections

# --- Cached: Load one section's DataFrame (only channels 1,3,4 if present) ---
def load_section_df(lines, columns, section_bounds):
    section_lines = lines[section_bounds[0]:section_bounds[1]]
    df = pd.read_csv(StringIO('\n'.join(section_lines)), sep='\t', header=None)
    df.columns = columns[:df.shape[1]]
    needed_cols = [col for col in ['Time (s)', 'Channel 1', 'Channel 3', 'Channel 4'] if col in df.columns]
    return df[needed_cols]

# Read zip file
def extract_zip(zip_bytes):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        for filename in z.namelist():
            if filename.lower().endswith('.txt'):
                with z.open(filename) as f:
                    return f.read()
    return None

# Detect peaks
@st.cache_data(show_spinner=False)
def cached_find_peaks(signal, freq, prominence, distance, detect_troughs):
    data_to_analyze = -signal if detect_troughs else signal
    peaks, _ = find_peaks(
        data_to_analyze,
        distance=distance * freq,
        prominence=prominence * np.std(signal)
    )
    return peaks

# MM:SS to seconds
def time_to_seconds(t):
    parts = t.strip().split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except:
        return 0
    return 0

# File Reading
uploaded_file = st.file_uploader("Upload LabChart Text or ZIP", type=["txt", "zip"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.name.endswith('.zip'):
        file_bytes = extract_zip(file_bytes)
        if file_bytes is None:
            st.error("No .txt file found inside ZIP.")
            st.stop()

    # Avoid running every time
    if 'lines' not in st.session_state or 'columns' not in st.session_state or 'sections' not in st.session_state or st.session_state.get('file_bytes', None) != file_bytes:
        st.session_state.file_bytes = file_bytes
        st.session_state.lines, st.session_state.columns, st.session_state.sections = split_sections(file_bytes)
        st.session_state.last_section_idx = None  # reset cached section df

    lines = st.session_state.lines
    columns = st.session_state.columns
    sections = st.session_state.sections

    section_labels = [f"Section {i+1}" for i in range(len(sections))]
    selected_section = st.sidebar.selectbox("Select Section", section_labels)
    section_idx = section_labels.index(selected_section)

    # Load section df once
    if st.session_state.get('last_section_idx') != section_idx:
        st.session_state.df = load_section_df(lines, columns, sections[section_idx])
        st.session_state.last_section_idx = section_idx

    df = st.session_state.df

    st.write(f"### Data Sample for {selected_section}")
    st.dataframe(df.head())

    show_fish1 = st.sidebar.checkbox("Show Channel 3 (Fish 1 EKG)", 'Channel 3' in df.columns)
    show_fish2 = st.sidebar.checkbox("Show Channel 4 (Fish 2 EKG)", 'Channel 4' in df.columns)
    detect_troughs = st.sidebar.checkbox("Detect Troughs")

    prominence = st.sidebar.slider("Peak Prominence Multiplier", 0.1, 7.0, 1.5, 0.1)
    distance = st.sidebar.slider("Minimum Peak Distance (s)", 0.1, 2.0, 0.3)

    time_vals = df['Time (s)'].values
    freq = 1 / (time_vals[1] - time_vals[0]) if len(time_vals) > 1 else 1000

    fig = go.Figure()
    cached_peaks = {}

    if show_fish1 and 'Channel 3' in df.columns:
        sig1 = df['Channel 3'].values
        peaks1 = cached_find_peaks(sig1, freq, prominence, distance, detect_troughs)
        cached_peaks['Fish 1'] = (time_vals[peaks1], sig1[peaks1])
        fig.add_trace(go.Scatter(x=time_vals, y=sig1, mode='lines', name='Fish 1', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=time_vals[peaks1], y=sig1[peaks1], mode='markers', name='Fish 1 Peaks', marker=dict(color='red', size=6)))

    if show_fish2 and 'Channel 4' in df.columns:
        sig2 = df['Channel 4'].values
        peaks2 = cached_find_peaks(sig2, freq, prominence, distance, detect_troughs)
        cached_peaks['Fish 2'] = (time_vals[peaks2], sig2[peaks2])
        fig.add_trace(go.Scatter(x=time_vals, y=sig2, mode='lines', name='Fish 2', line=dict(color='magenta')))
        fig.add_trace(go.Scatter(x=time_vals[peaks2], y=sig2[peaks2], mode='markers', name='Fish 2 Peaks', marker=dict(color='deepskyblue', size=6)))

    fig.update_layout(
        title=f"{selected_section} EKG Signals",
        xaxis_title='Time (MM:SS)',
        yaxis_title='Signal (µV)',
        hovermode='x unified',
        height=600,
        xaxis=dict(
            tickmode='array',
            tickvals=np.linspace(time_vals[0], time_vals[-1], 10),
            ticktext=[f"{int(t // 60)}:{int(t % 60):02d}" for t in np.linspace(time_vals[0], time_vals[-1], 10)],
            rangeslider=dict(visible=True)
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    # Peak stats
    st.write("## Peak Stats")
    start_time_entry = st.text_input("Start Time (MM:SS)", value="0:00")
    end_time_entry = st.text_input("End Time (MM:SS)", value=f"{int(time_vals[-1]//60)}:{int(time_vals[-1]%60):02d}")

    start_sec, end_sec = time_to_seconds(start_time_entry), time_to_seconds(end_time_entry)

    for fish_label, (peak_times, _) in cached_peaks.items():
        mask = (peak_times >= start_sec) & (peak_times <= end_sec)
        filtered_times = peak_times[mask]
        if len(filtered_times) > 1:
            rr_intervals = np.diff(filtered_times)
            bpm_vals = 60.0 / rr_intervals
            st.write(f"**{fish_label}**")
            st.write(f"- Number of Beats: {len(filtered_times)}")
            st.write(f"- Mean BPM: {np.mean(bpm_vals):.2f}")
            st.write(f"- Std Dev BPM: {np.std(bpm_vals):.2f}")
            st.write(f"- Max BPM: {np.max(bpm_vals):.2f}")
        else:
            st.write(f"**{fish_label}** — No sufficient peaks in range.")
