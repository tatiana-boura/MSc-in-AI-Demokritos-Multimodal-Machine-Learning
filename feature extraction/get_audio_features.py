import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")


def extract_mfcc_feature_metric(audio_file_name, signal, metrics, sample_rate, number_of_mfcc=13):
    mfcc_alt = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    mfcc_features = {"audio_path": audio_file_name}

    for key, value in metrics.items():

        metric, arg = value

        for i in range(0, number_of_mfcc):
            # mfcc coefficient
            key_name = f'mfcc_{i + 1}_{key}'
            mfcc_value = metric(mfcc_alt[i], arg)
            mfcc_features.update({key_name: mfcc_value})

            # mfcc delta coefficient
            key_name = f'mfcc_delta_{i + 1}_{key}'
            mfcc_value = metric(delta[i], arg)
            mfcc_features.update({key_name: mfcc_value})

            # mfcc accelerate coefficient
            key_name = f'mfcc_accelerate_{i + 1}_{key}'
            mfcc_value = metric(accelerate[i], arg)
            mfcc_features.update({key_name: mfcc_value})

    df = pd.DataFrame.from_records(data=[mfcc_features])
    return df


def extract_feature_metric(audio_file_path, audio_name, metrics):
    # 1. Importing 1 file
    y, sr = librosa.load(path=audio_file_path, sr=22050)

    signal = y

    # 2. Fourier Transform

    # Short-time Fourier transform (STFT)
    d_audio = np.abs(librosa.stft(signal))

    # 3. Spectrogram
    db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

    # 4. Create the Mel Spectrograms
    s_audio = librosa.feature.melspectrogram(y=signal, sr=sr)
    s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

    # 5. Zero crossings

    # 6. Harmonics and Perceptual
    y_harm, y_perc = librosa.effects.hpss(signal)

    # 7. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
    spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

    # 8. Chroma Frequencies
    chromagram = librosa.feature.chroma_stft(y=signal, sr=sr)

    # 9. Tempo BPM
    tempo_y, _ = librosa.beat.beat_track(y=signal, sr=sr)

    # 10. Spectral Roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]

    # 10. Spectral Flux
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)

    # Spectral Bandwidth
    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=3)[0]
    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y=signal, sr=sr, p=4)[0]

    audio_features = {"audio_path": audio_file_path,
                      "song_name" : audio_name,
                      "tempo_bpm": tempo_y
                      }

    for key, value in metrics.items():
        metric, arg = value

        audio_features.update({f'zero_crossing_rate_{key}': metric(librosa.feature.zero_crossing_rate(signal)[0], arg),
                               f'spectrogram_{key}': metric(db_audio[0], arg),
                               f'mel_spectrogram_{key}': metric(s_db_audio[0], arg),
                               f'harmonics_{key}': metric(y_harm, arg),
                               f'perceptual_shock_wave_{key}': metric(y_perc, arg),
                               f'spectral_centroids_{key}': metric(spectral_centroids, arg),
                               f'spectral_centroids_delta_{key}': metric(spectral_centroids_delta, arg),
                               f'spectral_centroids_accelerate_{key}': metric(spectral_centroids_accelerate, arg),
                               f'chroma1_{key}': metric(chromagram[0], arg),
                               f'chroma2_{key}': metric(chromagram[1], arg),
                               f'chroma3_{key}': metric(chromagram[2], arg),
                               f'chroma4_{key}': metric(chromagram[3], arg),
                               f'chroma5_{key}': metric(chromagram[4], arg),
                               f'chroma6_{key}': metric(chromagram[5], arg),
                               f'chroma7_{key}': metric(chromagram[6], arg),
                               f'chroma8_{key}': metric(chromagram[7], arg),
                               f'chroma9_{key}': metric(chromagram[8], arg),
                               f'chroma10_{key}': metric(chromagram[9], arg),
                               f'chroma11_{key}': metric(chromagram[10], arg),
                               f'chroma12_{key}': metric(chromagram[11], arg),
                               f'spectral_rolloff_{key}': metric(spectral_rolloff, arg),
                               f'spectral_flux_{key}': metric(onset_env, arg),
                               f'spectral_bandwidth_2_{key}': metric(spectral_bandwidth_2, arg),
                               f'spectral_bandwidth_3_{key}': metric(spectral_bandwidth_3, arg),
                               f'spectral_bandwidth_4_{key}': metric(spectral_bandwidth_4, arg),
                               })

    # extract mfcc features
    mfcc_df = extract_mfcc_feature_metric(audio_file_name=audio_file_path, signal=signal,
                                          metrics=metrics, sample_rate=sr)

    df = pd.DataFrame.from_records(data=[audio_features])

    df = pd.merge(df, mfcc_df, on='audio_path')

    return df


def get_audio_features(data, config):

    metrics = {"mean": [np.mean, None],
               "median": [np.median, None],
               "std": [np.std, None],
               "percentile": [np.percentile, 0.75]}

    for i in data.index:
    #for i in range(10):
        if (i + 1) % 100 == 0:
            print(f'Start preprocessing video-clip {i}...')

        audio_feats_df = extract_feature_metric(data['path_audio'][i], data['song_name'][i], metrics) if i == 0 else pd.concat(
            (audio_feats_df, extract_feature_metric(data['path_audio'][i], data['song_name'][i], metrics)), ignore_index=True)

    audio_feats_df.to_csv("./audio_feats.csv", index=False)

    return audio_feats_df
