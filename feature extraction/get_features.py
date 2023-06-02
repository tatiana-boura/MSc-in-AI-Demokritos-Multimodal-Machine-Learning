import pandas as pd
import yaml
import os

from get_audio_features import get_audio_features
from get_video_features import get_video_features


def create_index(config):

    audio_files = pd.DataFrame(columns=["path", "genre", "song_name"])
    video_files = pd.DataFrame(columns=["path", "genre", "song_name"])

    for path in config:
        for filename in os.listdir(path):
            row = {"path": os.path.join(path, filename), "genre": path.split("\\\\")[-1], "song_name": filename[:-9]}
            if filename.endswith(".m4a"):
                audio_files.loc[len(audio_files)] = row
            elif filename.endswith(".mp4"):
                video_files.loc[len(video_files)] = row

    index = pd.merge(audio_files, video_files, on="song_name", how="inner", suffixes=("_audio", "_video")).drop(
        "genre_video", axis=1).rename(columns={'genre_audio': 'genre'})

    return index


def get_features():

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    BLUES_PATH = config["BLUES_PATH"]
    COUNTRY_PATH = config["COUNTRY_PATH"]
    HIP_HOP_PATH = config["HIP_HOP_PATH"]
    INDIE_PATH = config["INDIE_PATH"]
    METAL_PATH = config["METAL_PATH"]
    POP_ROCK_PATH = config["POP_ROCK_PATH"]
    PUNK_PATH = config["PUNK_PATH"]
    SOUL_PATH = config["SOUL_PATH"]
    DRILL_PATH= config["DRILL_PATH"]

    clip_config = [BLUES_PATH, COUNTRY_PATH, HIP_HOP_PATH, INDIE_PATH, METAL_PATH, POP_ROCK_PATH, PUNK_PATH, SOUL_PATH, DRILL_PATH]

    index = create_index(config=clip_config)
    '''
    print(index)
    '''
    audio_feats = get_audio_features(index[["path_audio", "song_name", "genre"]], config)

    print(audio_feats)
    '''

    video_feats = get_video_features(index[["path_video", "song_name", "genre"]], config)
    print(video_feats)
    '''


get_features()
