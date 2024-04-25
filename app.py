# app.py

import streamlit as st
import neurokit2 as nk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

def main():
    # Streamlitアプリケーションのタイトルを設定
    st.title('ECG Analysis App')

    # ユーザーが設定できる各パラメータのスライダーを追加
    # パラメータ設定
    duration = st.sidebar.slider("Duration", min_value=100, max_value=2000, value=900, step=100)
    sampling_rate = st.sidebar.slider("Sampling Rate", min_value=200, max_value=1000, value=200, step=100)
    heart_rate = st.sidebar.slider("Heart Rate", min_value=40, max_value=150, value=70, step=10)
    heart_rate_std = st.sidebar.slider("Heart Rate Standard Deviation", min_value=0, max_value=10, value=2, step=1)
    noise = st.sidebar.slider("Noise", min_value=0.01, max_value=5.0, value=0.1, step=0.01)
    Noise_range = st.sidebar.slider("Noise Range", min_value=1, max_value=20, value=10, step=1)
    Noise_max = st.sidebar.slider("Noise Max", min_value=12, max_value=30, value=24, step=1)

    # シミュレートされた心電図の生成
    simulated_ecg = nk.ecg_simulate(duration=duration, sampling_rate=sampling_rate, heart_rate=heart_rate, heart_rate_std=heart_rate_std, noise=noise, method='"multileads"')
    ecg_df = pd.DataFrame(simulated_ecg, columns=["ECG"])

    # ECG信号の前処理とピーク検出を行う
    signals, info = nk.ecg_process(ecg_df["ECG"], sampling_rate=sampling_rate)

    # ピークのインデックスを取得
    r_peak_indices_ms = (info['ECG_R_Peaks'] / 1) * 1

    # ピーク間の時間差を計算
    peak_diff = np.diff(r_peak_indices_ms)

    #=====================================================
    #
    #=====================================================

    # peak_diffにr_peak_indices_msの最初の値を挿入する
    peak_diff_with_time = np.insert(peak_diff, 0, r_peak_indices_ms[0])

    # HRVの振幅変数を作成する
    hrv_series = pd.DataFrame({
        "Time": r_peak_indices_ms,
        "Amplitude": peak_diff_with_time,
    })

    HRV_sampling_rate = len(peak_diff_with_time)/duration

    new_sampling_rate = len(peak_diff_with_time)/duration*2  # 新しいサンプリングレート
    new_time = np.linspace(hrv_series["Time"].min(), hrv_series["Time"].max(), int(len(hrv_series) * (new_sampling_rate / HRV_sampling_rate)))

    # スプライン補間を行い、等間隔のデータ系列を作成
    spline_interpolator = interp1d(hrv_series["Time"], hrv_series["Amplitude"], kind='cubic')
    interpolated_amplitude = spline_interpolator(new_time)

    # 補間されたデータをDataFrameに格納
    interpolated_data = pd.DataFrame({
        "Time": new_time,
        "Amplitude": interpolated_amplitude
    })


    # 変更後のピーク間の時間差を計算
    modified_peak_diff = peak_diff + np.where(np.random.rand(len(peak_diff)) < 0.8,
                                               np.random.randint(-Noise_range, Noise_range, size=len(peak_diff)),  # 絶対値Noise_range以内の値を生成
                                               np.random.choice(np.concatenate((np.arange(-Noise_max, -10), np.arange(10, Noise_max))), size=len(peak_diff)))

    # modified_peak_diffにr_peak_indices_msの最初の値を挿入する
    modified_peak_diff_with_time = np.insert(modified_peak_diff, 0, r_peak_indices_ms[0])

    # 時間系列を格納する空列を作る
    cumulative_sum_series = []

    for i in range(len(r_peak_indices_ms)):
        if i == 0:
            cumulative_sum_series.append(r_peak_indices_ms[0])
        else:
            cumulative_sum_series.append(cumulative_sum_series[i-1]+modified_peak_diff[i-1])

    # HRVの振幅変数を作成する
    modified_hrv_series = pd.DataFrame({
        "Time": cumulative_sum_series,
        "Amplitude": modified_peak_diff_with_time,
    })
    #==================================================
    #splineしてHRV作成
    #==================================================
    # サンプリング周波数を計算
    #sampling_rate = len(modified_hrv_series)/duration

    # サンプリングレートを増やすための新しい時間軸を生成
    #new_sampling_rate = 3  # 新しいサンプリングレート
    modified_new_time = np.linspace(modified_hrv_series["Time"].min(), modified_hrv_series["Time"].max(), int(len(modified_hrv_series) * (new_sampling_rate / HRV_sampling_rate)))


    # スプライン補間を行い、等間隔のデータ系列を作成
    modified_spline_interpolator = interp1d(modified_hrv_series["Time"], modified_hrv_series["Amplitude"], kind='cubic')
    modified_interpolated_amplitude = modified_spline_interpolator(modified_new_time)

    # 補間されたデータをDataFrameに格納
    modified_interpolated_data = pd.DataFrame({
        "Time": modified_new_time,
        "Amplitude": modified_interpolated_amplitude
    })


    # FFTを計算する
    sampling_rate_peak_diff = len(interpolated_amplitude) / duration
    fft_peak_diff = np.fft.fft(interpolated_amplitude)
    fft_modified_peak_diff = np.fft.fft(modified_interpolated_amplitude)
    freq = np.fft.fftfreq(len(interpolated_amplitude), 1/sampling_rate_peak_diff)


    lf_start_freq = 0.04  # LF帯域の開始周波数
    lf_end_freq = 0.15    # LF帯域の終了周波数

    # LF帯域の開始周波数に対応するインデックスを取得
    lf_start_index = np.argmax(freq >= lf_start_freq)

    # LF帯域の終了周波数に対応するインデックスを取得
    lf_end_index = np.argmax(freq >= lf_end_freq)

    # LF帯域の振幅スペクトルを抽出する
    lf_fft_peak_diff = fft_peak_diff[lf_start_index:lf_end_index]
    lf_fft_modified_peak_diff = fft_modified_peak_diff[lf_start_index:lf_end_index]

    # 振幅スペクトルの間隔
    lf_dx = freq[1] - freq[0]

    # 台形則を用いて積分を計算する
    lf_power = trapz(np.abs(lf_fft_peak_diff), dx=lf_dx)
    lf_power_modified = trapz(np.abs(lf_fft_modified_peak_diff), dx=lf_dx)

    hf_start_freq = 0.15  # LF帯域の開始周波数
    hf_end_freq = 0.4    # LF帯域の終了周波数

    # LF帯域の開始周波数に対応するインデックスを取得
    hf_start_index = np.argmax(freq >= hf_start_freq)

    # LF帯域の終了周波数に対応するインデックスを取得
    hf_end_index = np.argmax(freq >= hf_end_freq)

    # LF帯域の振幅スペクトルを抽出する
    hf_fft_peak_diff = fft_peak_diff[hf_start_index:hf_end_index]
    hf_fft_modified_peak_diff = fft_modified_peak_diff[hf_start_index:hf_end_index]

    # 振幅スペクトルの間隔
    hf_dx = freq[1] - freq[0]

    # 台形則を用いて積分を計算する
    hf_power = trapz(np.abs(hf_fft_peak_diff), dx=hf_dx)
    hf_power_modified = trapz(np.abs(hf_fft_modified_peak_diff), dx=hf_dx)

    lf_hf_ratio = lf_power/hf_power
    lf_hf_ratio_modified = lf_power_modified/hf_power_modified

    # プロット
    # プロット範囲を0.01Hzから1Hzに制限
    freq_range_mask = (freq >= 0.01) & (freq <= 1)

    # プロット
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # プロット範囲内の周波数成分のみをプロット
    ax1.plot(freq[freq_range_mask], np.abs(fft_peak_diff[freq_range_mask]), label='Standard HRV', color='blue')
    ax1.plot(freq[freq_range_mask], np.abs(fft_modified_peak_diff[freq_range_mask]), label='Estimated HRV', color='red')

    # ラベルなどの設定
    ax1.set_title('HRV FFT')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True)

    # 0.01から1Hzまでの成分を含むデータを取得
    freq_range_mask = (freq >= 0.01) & (freq <= 1)

    # プロットを作成する（LF/HF比率）
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    bar_colors = ['blue', 'red']
    ax2.bar(['Standard', 'Estimated'], [lf_hf_ratio, lf_hf_ratio_modified], color=bar_colors, alpha=0.5, label='LF/HF Ratio')
    ax2.set_title('LF/HF Ratio')
    ax2.set_ylabel('Ratio')
    ax2.legend()
    ax2.grid(True)

    # プロットを作成する（ピーク間の時間差のデータ）
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(peak_diff, label='Standard ECG', color='blue')
    ax3.plot(modified_peak_diff, label='Estimated HRV', color='red')
    ax3.set_title('Peak Interval Data')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Peak Interval')
    ax3.legend()
    ax3.grid(True)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.plot(np.arange(len(ecg_df["ECG"]))[:1500] / 200, ecg_df["ECG"][:1500], label='ECG', color='green')
    ax4.set_title('HRV')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(True)

    # グラフを表示
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)

if __name__ == "__main__":
    main()
