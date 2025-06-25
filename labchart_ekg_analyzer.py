import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from io import StringIO
import zipfile
import plotly.graph_objects as go
import tempfile
import os

st.set_page_config(layout="wide")
st.title("LabChart EKG Analyzer")

def extract_zip(file_path):
    if file_path.lower().endswith(".zip"):
        with zipfile.ZipFile(file_path) as z:
            for fn in z.namelist():
                if fn.lower().endswith('.txt'):
                    with z.open(fn) as extracted_file, tempfile.NamedTemporaryFile(delete=False) as tmp_extracted:
                        for chunk in extracted_file:
                            tmp_extracted.write(chunk)
                        return tmp_extracted.name
        return None
    return file_path

@st.cache_data(show_spinner=False)
def split_sections(file_path):
    columns, sections = None, []
    prev_time = None
    section_start = None

    with open(file_path, 'rb') as f:
        while True:
            start_pos = f.tell()
            line = f.readline()
            if not line:
                break

            try:
                decoded_line = line.decode('ISO-8859-1')
            except:
                continue

            if decoded_line.startswith("ChannelTitle="):
                titles = decoded_line.replace("ChannelTitle=", "").strip().split('\t')
                columns = ['Time (s)'] + titles

            if decoded_line and (decoded_line[0].isdigit() or decoded_line[0] == '.'):
                try:
                    t = float(decoded_line.split('\t', 1)[0])
                    if section_start is None:
                        section_start = start_pos
                    elif prev_time is not None and t < prev_time:
                        sections.append((section_start, start_pos))
                        section_start = start_pos
                    prev_time = t
                except:
                    continue

    if section_start is not None:
        with open(file_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            sections.append((section_start, f.tell()))

    return columns, sections

def load_section_df(file_path, columns, section_bounds):
    start, end = section_bounds
    with open(file_path, 'rb') as f:
        f.seek(start)
        section_bytes = f.read(end - start)
    section_text = section_bytes.decode('ISO-8859-1')
    df = pd.read_csv(StringIO(section_text), sep='\t', header=None)
    df.columns = columns[:df.shape[1]]
    keep_cols = [c for c in ['Time (s)', 'Channel 1', 'Channel 3', 'Channel 4'] if c in df.columns]
    df = df[keep_cols]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.dropna(subset=['Time (s)'])

@st.cache_data(show_spinner=False)
def cached_find_peaks(signal, freq, prom, dist, detect_troughs):
    data = -signal if detect_troughs else signal
    peaks, _ = find_peaks(data, distance=dist*freq, prominence=prom*np.std(signal))
    return peaks

def time_to_seconds(s):
    parts = s.strip().split(':')
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 1:
            return int(parts[0])
    except:
        return 0
    return 0

uploaded_file = st.file_uploader("Upload LabChart Text or ZIP", type=["txt", "zip"])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        for chunk in uploaded_file:
            tmp_file.write(chunk)
        uploaded_path = tmp_file.name

    file_path = extract_zip(uploaded_path)
    if file_path is None:
        st.error("No .txt file found inside ZIP.")
        st.stop()

    if 'columns' not in st.session_state or st.session_state.get('file_path') != file_path:
        st.session_state.file_path = file_path
        st.session_state.columns, st.session_state.sections = split_sections(file_path)
        st.session_state.last_section_idx = None

    columns, sections = st.session_state.columns, st.session_state.sections
    section_labels = [f"Section {i+1}" for i in range(len(sections))]
    selected_section = st.sidebar.selectbox("Pick Section", section_labels)
    section_idx = section_labels.index(selected_section)

    if st.session_state.get('last_section_idx') != section_idx:
        st.session_state.df = load_section_df(file_path, columns, sections[section_idx])
        st.session_state.last_section_idx = section_idx

    df = st.session_state.df
    st.dataframe(df.head())

    show_fish1 = st.sidebar.checkbox("Show Channel 3 (Fish 1 EKG)", 'Channel 3' in df.columns)
    show_fish2 = st.sidebar.checkbox("Show Channel 4 (Fish 2 EKG)", 'Channel 4' in df.columns)
    detect_troughs = st.sidebar.checkbox("Detect Troughs")

    prom = st.sidebar.slider("Peak Prominence Multiplier", 0.1, 7.0, 1.5, 0.1)
    dist = st.sidebar.slider("Minimum Peak Distance (s)", 0.1, 2.0, 0.3)

    times = df['Time (s)'].values
    freq = 1 / (times[1] - times[0]) if len(times) > 1 else 1000

    fig = go.Figure()
    cached_peaks = {}

    if show_fish1 and 'Channel 3' in df.columns:
        sig1 = df['Channel 3'].values
        peaks1 = cached_find_peaks(sig1, freq, prom, dist, detect_troughs)
        cached_peaks['Fish 1'] = (times[peaks1], sig1[peaks1])
        fig.add_trace(go.Scatter(x=times, y=sig1, mode='lines', name='Fish 1', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=times[peaks1], y=sig1[peaks1], mode='markers', name='Fish 1 Peaks', marker=dict(color='red', size=6)))

    if show_fish2 and 'Channel 4' in df.columns:
        sig2 = df['Channel 4'].values
        peaks2 = cached_find_peaks(sig2, freq, prom, dist, detect_troughs)
        cached_peaks['Fish 2'] = (times[peaks2], sig2[peaks2])
        fig.add_trace(go.Scatter(x=times, y=sig2, mode='lines', name='Fish 2', line=dict(color='magenta')))
        fig.add_trace(go.Scatter(x=times[peaks2], y=sig2[peaks2], mode='markers', name='Fish 2 Peaks', marker=dict(color='deepskyblue', size=6)))

    ticks = np.linspace(times[0], times[-1], 10)
    fig.update_layout(
        title=f"{selected_section} EKG Signals",
        xaxis=dict(
            tickmode='array',
            tickvals=ticks,
            ticktext=[f"{int(t // 60)}:{int(t % 60):02d}" for t in ticks],
            rangeslider=dict(visible=True),
        ),
        yaxis_title='Signal (µV)',
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("## Peak Stats")
    start_time = st.text_input("Start (MM:SS)", "0:00")
    end_time = st.text_input("End (MM:SS)", f"{int(times[-1]//60)}:{int(times[-1]%60):02d}")
    start_sec, end_sec = time_to_seconds(start_time), time_to_seconds(end_time)

    for label, (peak_times, _) in cached_peaks.items():
        mask = (peak_times >= start_sec) & (peak_times <= end_sec)
        filtered_times = peak_times[mask]
        if len(filtered_times) > 1:
            rr = np.diff(filtered_times)
            bpm_vals = 60.0 / rr
            st.write(f"**{label}**")
            st.write(f"- Number of Beats: {len(filtered_times)}")
            st.write(f"- Mean BPM: {np.mean(bpm_vals):.2f}")
            st.write(f"- Std Dev BPM: {np.std(bpm_vals):.2f}")
            st.write(f"- Max BPM: {np.max(bpm_vals):.2f}")
        else:
            st.write(f"**{label}** — No sufficient peaks in range.")