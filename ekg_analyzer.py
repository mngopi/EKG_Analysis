# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np
from scipy.signal import find_peaks
st.set_page_config(layout="wide")
st.title("LabChart EKG Analyzer")


# File uploader replaces tkinter file dialog
uploaded_file = st.file_uploader("Select LabChart Text File", type=["txt"])

if uploaded_file is not None:
    # Read lines from uploaded file
    lines = uploaded_file.getvalue().decode('ISO-8859-1').splitlines()

    # Identify data start lines
    data_starts = [i for i, line in enumerate(lines) if line.strip() and (line[0].isdigit() or line[0] == '.')]

    # Split sections by time reset
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

    # Read each section to DataFrame
    dfs = []
    for sec in data_sections:
        section_lines = lines[sec[0]:sec[1]]
        df = pd.read_csv(StringIO('\n'.join(section_lines)), sep='\t', header=None)
        df.columns = ['Time (s)', 'Date', 'Channel 1', 'Channel 2', 'Channel 3', 'Channel 4', 'Channel 5', 'Channel 6']
        dfs.append(df)

    df_pre = dfs[0]
    df_post = dfs[1]

    st.write("### Pre-injection Data Sample")
    st.dataframe(df_pre.head())

    st.write("### Post-injection Data Sample")
    st.dataframe(df_post.head())

    # Sidebar inputs
    phase = st.sidebar.selectbox("Select phase", ['Pre-Injection', 'Post-Injection'])
    if phase == 'Pre-Injection':
        df_selected = df_pre
    else:
        df_selected = df_post
    start = st.sidebar.number_input("Start time (seconds)", min_value=0.0, max_value=float(df_selected['Time (s)'].max()), value=60.0)
    window_size = st.sidebar.slider("Window size (samples)", min_value=1000, max_value=100000, value=10000, step=500)
    show_fish1 = st.sidebar.checkbox("Show Fish 1 EKG", value=True)
    show_fish2 = st.sidebar.checkbox("Show Fish 2 EKG", value=True)
    prominence = st.sidebar.slider("Peak prominence multiplier", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

    # Plot function
    def plot_window(phase, start, window_size, show_fish1, show_fish2, prominence):
        if phase == 'Pre-Injection':
            df_selected = df_pre
        else:
            df_selected = df_post

        time_vals = df_selected['Time (s)'].values
        start_idx = (np.abs(time_vals - start)).argmin()
        max_start_idx = max(0, len(df_selected) - window_size)
        start_idx = min(start_idx, max_start_idx)

        plt.figure(figsize=(20, 8))
        time_window = time_vals[start_idx:start_idx+window_size]

        time_diff = time_vals[1] - time_vals[0]
        freq = 1 / time_diff

        if show_fish1:
            signal_fish1 = df_selected['Channel 3'].iloc[start_idx:start_idx+window_size].values
            plt.plot(time_window, signal_fish1, label='Fish 1 EKG', color='darkgreen')
            peaks_fish1, _ = find_peaks(signal_fish1, distance=0.3*freq, prominence=prominence*np.std(signal_fish1))
            plt.plot(time_window[peaks_fish1], signal_fish1[peaks_fish1], 'o', color='crimson', label='Fish 1 Peaks')

        if show_fish2:
            signal_fish2 = df_selected['Channel 4'].iloc[start_idx:start_idx+window_size].values
            plt.plot(time_window, signal_fish2, label='Fish 2 EKG', color='magenta')
            peaks_fish2, _ = find_peaks(signal_fish2, distance=0.3*freq, prominence=prominence*np.std(signal_fish2))
            plt.plot(time_window[peaks_fish2], signal_fish2[peaks_fish2], 'o', color='deepskyblue', label='Fish 2 Peaks')

        plt.xlabel('Time (s)')
        plt.ylabel('EKG Signal (µV)')
        plt.title(f"{phase} EKG Signals")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.close()

    plot_window(phase, start, window_size, show_fish1, show_fish2, prominence)

    # Peak stats inputs
    if phase == 'Pre-Injection':
        df_selected = df_pre
    else:
        df_selected = df_post
    st.write("## Peak Statistics")
    start_stat = st.number_input("Start time for stats (s)", min_value=0.0, max_value=float(df_selected['Time (s)'].max()), value=60.0)
    end_stat = st.number_input("End time for stats (s)", min_value=0.0, max_value=float(df_selected['Time (s)'].max()), value=120.0)
    #prominence_stat = st.slider("Peak prominence multiplier for stats", min_value=0.1, max_value=5.0, value=1.5, step=0.1)

    def get_peak_stats(phase, start, end, prominence):
        if phase == 'Pre-Injection':
            df_selected = df_pre
        else:
            df_selected = df_post

        df_window = df_selected[(df_selected['Time (s)'] >= start) & (df_selected['Time (s)'] < end)]
        time_vals = df_window['Time (s)'].values
        time_diff = time_vals[1] - time_vals[0]
        freq = 1 / time_diff

        results = []

        signal1 = df_window['Channel 3'].values
        peaks1, _ = find_peaks(signal1, distance=0.3*freq, prominence=prominence*np.std(signal1)) # 0.3 seconds between peaks = 200 bpm max
        if len(peaks1) > 1:
            peak_times1 = time_vals[peaks1] # find times of crimson points
            rr_intervals1 = np.diff(peak_times1) # difference between consecutive peaks in seconds
            bpm_values1 = 60.0 / rr_intervals1 # 60 / seconds per beat, or BPM

            results.append({
                'Fish': 'Fish 1',
                'Num Beats': len(peaks1),
                'Mean BPM': np.mean(bpm_values1),
                'Std Dev BPM': np.std(bpm_values1),
                'Max BPM': np.max(bpm_values1)
            })

        signal2 = df_window['Channel 4'].values
        peaks2, _ = find_peaks(signal2, distance=0.3*freq, prominence=prominence*np.std(signal2))
        if len(peaks2) > 1:
            peak_times2 = time_vals[peaks2]
            rr_intervals2 = np.diff(peak_times2)
            bpm_values2 = 60.0 / rr_intervals2

            results.append({
                'Fish': 'Fish 2',
                'Num Beats': len(peaks2),
                'Mean BPM': np.mean(bpm_values2),
                'Std Dev BPM': np.std(bpm_values2),
                'Max BPM': np.max(bpm_values2)
            })

        return results

    stats = get_peak_stats(phase, start_stat, end_stat, prominence)
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
