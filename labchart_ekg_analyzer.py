import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from io import StringIO, BytesIO
import zipfile

st.set_page_config(layout="wide")
st.title("LabChart EKG Analyzer")

@st.cache_data
def parse_labchart_file(file_bytes):
    # file_bytes is bytes of .txt file content
    lines = file_bytes.decode('ISO-8859-1').splitlines()

    channel_title_line = next(line for line in lines if line.startswith("ChannelTitle="))
    channel_titles = channel_title_line.replace("ChannelTitle=", "").strip().split('\t')
    columns = ['Time (s)'] + channel_titles

    data_starts = [i for i, line in enumerate(lines) if line.strip() and (line[0].isdigit() or line[0] == '.')]
    data_sections = []
    current_section = [data_starts[0]]
    for i in range(1, len(data_starts)):
        prev_line = lines[data_starts[i-1]].split('\t')[0]
        this_line = lines[data_starts[i]].split('\t')[0]
        try:
            if float(this_line) < float(prev_line):
                current_section.append(data_starts[i-1]+1)
                data_sections.append(current_section)
                current_section = [data_starts[i]]
        except:
            continue
    current_section.append(len(lines))
    data_sections.append(current_section)

    dfs = []
    for sec in data_sections:
        section_lines = lines[sec[0]:sec[1]]
        df = pd.read_csv(StringIO('\n'.join(section_lines)), sep='\t', header=None)
        df.columns = columns[:df.shape[1]]
        dfs.append(df)

    return dfs

@st.cache_data
def find_peaks_channel(signal, freq, prominence, distance, detect_troughs):
    data_to_analyze = -signal if detect_troughs else signal
    peaks, _ = find_peaks(
        data_to_analyze,
        distance=distance * freq,
        prominence=prominence * np.std(signal)
    )
    return peaks

@st.cache_data
def compute_peak_stats(peaks, time_vals_sec, start_stat, end_stat):
    peak_times = time_vals_sec[peaks]
    mask = (peak_times >= start_stat) & (peak_times < end_stat)
    filtered_peaks = peaks[mask]

    if len(filtered_peaks) > 1:
        rr_intervals = np.diff(time_vals_sec[filtered_peaks])
        bpm_values = 60.0 / rr_intervals
        return {
            'Num Beats': len(filtered_peaks),
            'Mean BPM': np.mean(bpm_values),
            'Std Dev BPM': np.std(bpm_values),
            'Max BPM': np.max(bpm_values)
        }
    else:
        return None

def time_to_seconds(time):
    parts = time.strip().split(':')
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 1:
            return int(parts[0])
        else:
            return 0
    except:
        return 0

uploaded_file = st.file_uploader(
    "Select LabChart Text File or ZIP containing it",
    type=["txt", "zip"]
)

def extract_txt_from_zip(zip_bytes):
    with zipfile.ZipFile(BytesIO(zip_bytes)) as z:
        # Find first txt file in zip
        for filename in z.namelist():
            if filename.lower().endswith('.txt'):
                with z.open(filename) as f:
                    return f.read()
    return None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    if uploaded_file.type == "application/zip" or uploaded_file.name.endswith('.zip'):
        file_bytes = extract_txt_from_zip(file_bytes)
        if file_bytes is None:
            st.error("No .txt file found inside the uploaded ZIP.")
            st.stop()
    # else assume file_bytes is the txt file content directly

    dfs = parse_labchart_file(file_bytes)

    if len(dfs) < 2:
        st.error("File does not contain enough data sections (need at least Pre and Post).")
    else:
        df_pre, df_post = dfs[0], dfs[1]

        st.write("### Pre-injection Data Sample")
        st.dataframe(df_pre.head())

        st.write("### Post-injection Data Sample")
        st.dataframe(df_post.head())

        phase = st.sidebar.selectbox("Select phase", ['Pre-Injection', 'Post-Injection'])
        df_selected = df_pre if phase == 'Pre-Injection' else df_post

        show_fish1 = st.sidebar.checkbox("Show Channel 3 EKG", value='Channel 3' in df_selected.columns)
        show_fish2 = st.sidebar.checkbox("Show Channel 4 EKG", value='Channel 4' in df_selected.columns)

        detect_troughs = st.sidebar.checkbox("Detect troughs")

        prominence = st.sidebar.slider("Peak prominence multiplier", 0.1, 7.0, 1.5, 0.1)
        distance = st.sidebar.slider("Minimum peak distance (seconds)", 0.1, 2.0, 0.3)

        time_vals_sec = df_selected['Time (s)'].values
        freq = 1 / (time_vals_sec[1] - time_vals_sec[0]) if len(time_vals_sec) > 1 else 1000

        cached_peaks = {}

        if show_fish1 and 'Channel 3' in df_selected.columns:
            sig1 = df_selected['Channel 3'].values
            cached_peaks['Fish 1'] = find_peaks_channel(sig1, freq, prominence, distance, detect_troughs)
        if show_fish2 and 'Channel 4' in df_selected.columns:
            sig2 = df_selected['Channel 4'].values
            cached_peaks['Fish 2'] = find_peaks_channel(sig2, freq, prominence, distance, detect_troughs)

        fig = go.Figure()

        if show_fish1 and 'Channel 3' in df_selected.columns:
            fig.add_trace(go.Scatter(x=time_vals_sec, y=sig1, mode='lines', name='Fish 1 EKG (Channel 3)', line=dict(color='darkgreen')))
            peaks1 = cached_peaks.get('Fish 1', [])
            fig.add_trace(go.Scatter(x=time_vals_sec[peaks1], y=sig1[peaks1], mode='markers', name='Fish 1 Peaks', marker=dict(color='crimson', size=6)))

        if show_fish2 and 'Channel 4' in df_selected.columns:
            fig.add_trace(go.Scatter(x=time_vals_sec, y=sig2, mode='lines', name='Fish 2 EKG (Channel 4)', line=dict(color='magenta')))
            peaks2 = cached_peaks.get('Fish 2', [])
            fig.add_trace(go.Scatter(x=time_vals_sec[peaks2], y=sig2[peaks2], mode='markers', name='Fish 2 Peaks', marker=dict(color='deepskyblue', size=6)))

        fig.update_layout(
            title=f"{phase} EKG Signals",
            xaxis_title='Time (MM:SS)',
            yaxis_title='EKG Signal (µV)',
            hovermode='x unified',
            height=600,
            xaxis=dict(
                tickmode='array',
                tickvals=np.linspace(0, time_vals_sec[-1], 10),
                ticktext=[f"{int(t//60)}:{int(t%60):02d}" for t in np.linspace(0, time_vals_sec[-1], 10)],
                rangeslider=dict(visible=True)
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write("## Peak Statistics")

        start_time_entry = st.text_input("Start time for stats (MM:SS)", value="1:00")
        end_time_entry = st.text_input("End time for stats (MM:SS)", value="2:00")

        start_stat = time_to_seconds(start_time_entry)
        end_stat = time_to_seconds(end_time_entry)

        stats_results = {}
        for fish_label, peaks in cached_peaks.items():
            stats = compute_peak_stats(peaks, time_vals_sec, start_stat, end_stat)
            if stats:
                stats_results[fish_label] = stats

        if stats_results:
            st.write("### Peak Stats Summary")
            for fish_label, stats in stats_results.items():
                st.write(f"**{fish_label}**")
                st.write(f"- Number of Beats: {stats['Num Beats']}")
                st.write(f"- Mean BPM: {stats['Mean BPM']:.4f}")
                st.write(f"- Std Dev BPM: {stats['Std Dev BPM']:.4f}")
                st.write(f"- Max BPM: {stats['Max BPM']:.4f}")
        else:
            st.info("No peaks detected in selected range.")