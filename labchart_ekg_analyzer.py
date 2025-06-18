import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from io import StringIO

st.title("LabChart EKG Analyzer")

@st.cache_data
def parse_labchart_file(uploaded_file):
    lines = uploaded_file.getvalue().decode('ISO-8859-1').splitlines()

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

uploaded_file = st.file_uploader("Select LabChart Text File", type=["txt"])

if uploaded_file is not None:
    dfs = parse_labchart_file(uploaded_file)
    df_pre, df_post = dfs[0], dfs[1]

    st.write("### Pre-injection Data Sample")
    st.dataframe(df_pre.head())

    st.write("### Post-injection Data Sample")
    st.dataframe(df_post.head())

    phase = st.sidebar.selectbox("Select phase", ['Pre-Injection', 'Post-Injection'])
    df_selected = df_pre if phase == 'Pre-Injection' else df_post

    max_time = df_selected['Time (s)'].max()

    show_fish1 = st.sidebar.checkbox("Show Channel 3 EKG", value='Channel 3' in df_selected.columns)
    show_fish2 = st.sidebar.checkbox("Show Channel 4 EKG", value='Channel 4' in df_selected.columns)

    detect_troughs = st.sidebar.checkbox("Detect troughs")

    prominence = st.sidebar.slider("Peak prominence multiplier", min_value=0.1, max_value=7.0, value=1.5, step=0.1)
    distance = st.sidebar.slider("Minimum peak distance (seconds)", min_value=0.1, max_value=2.0, value=0.3)

    time_vals_sec = df_selected['Time (s)'].values
    if len(time_vals_sec) > 1:
        time_diff = time_vals_sec[1] - time_vals_sec[0]
        freq = 1 / time_diff
    else:
        freq = 1000  # fallback

    def find_peaks_channel(signal, freq):
        data_to_analyze = -signal if detect_troughs else signal
        return find_peaks(
            data_to_analyze,
            distance=distance * freq,
            prominence=prominence * np.std(signal)
        )[0]

    # Find peaks for whole channel
    cached_peaks = {}
    if show_fish1 and 'Channel 3' in df_selected.columns:
        sig1 = df_selected['Channel 3'].values
        cached_peaks['Fish 1'] = find_peaks_channel(sig1, freq)
    if show_fish2 and 'Channel 4' in df_selected.columns:
        sig2 = df_selected['Channel 4'].values
        cached_peaks['Fish 2'] = find_peaks_channel(sig2, freq)

    # Plot
    fig = go.Figure()

    if show_fish1 and 'Channel 3' in df_selected.columns:
        fig.add_trace(go.Scatter(
            x=time_vals_sec, y=sig1, mode='lines',
            name='Fish 1 EKG (Channel 3)', line=dict(color='darkgreen')
        ))
        peaks1 = cached_peaks.get('Fish 1', [])
        fig.add_trace(go.Scatter(
            x=time_vals_sec[peaks1], y=sig1[peaks1], mode='markers',
            name='Fish 1 Peaks', marker=dict(color='crimson', size=6, symbol='circle')
        ))

    if show_fish2 and 'Channel 4' in df_selected.columns:
        fig.add_trace(go.Scatter(
            x=time_vals_sec, y=sig2, mode='lines',
            name='Fish 2 EKG (Channel 4)', line=dict(color='magenta')
        ))
        peaks2 = cached_peaks.get('Fish 2', [])
        fig.add_trace(go.Scatter(
            x=time_vals_sec[peaks2], y=sig2[peaks2], mode='markers',
            name='Fish 2 Peaks', marker=dict(color='deepskyblue', size=6, symbol='circle')
        ))

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
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("## Peak Statistics")

    start_time_entry = st.text_input("Start time for stats (MM:SS)", value="1:00")
    end_time_entry = st.text_input("End time for stats (MM:SS)", value="2:00")

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

    start_stat = time_to_seconds(start_time_entry)
    end_stat = time_to_seconds(end_time_entry)

    def get_peak_stats():
        results = []

        for fish_label, peaks in cached_peaks.items():
            # Filter peaks within selected time window
            peak_times = time_vals_sec[peaks]
            mask = (peak_times >= start_stat) & (peak_times < end_stat)
            filtered_peaks = peaks[mask]

            if len(filtered_peaks) > 1:
                rr_intervals = np.diff(time_vals_sec[filtered_peaks])
                bpm_values = 60.0 / rr_intervals
                results.append({
                    'Fish': fish_label,
                    'Num Beats': len(filtered_peaks),
                    'Mean BPM': np.mean(bpm_values),
                    'Std Dev BPM': np.std(bpm_values),
                    'Max BPM': np.max(bpm_values)
                })

        return results

    stats = get_peak_stats()
    if stats:
        st.write("### Peak Stats Summary")
        for result in stats:
            st.write(f"**{result['Fish']}**")
            st.write(f"- Number of Beats: {result['Num Beats']}")
            st.write(f"- Mean BPM: {result['Mean BPM']:.4f}")
            st.write(f"- Std Dev BPM: {result['Std Dev BPM']:.4f}")
            st.write(f"- Max BPM: {result['Max BPM']:.4f}")
    else:
        st.info("No peaks detected in selected range.")