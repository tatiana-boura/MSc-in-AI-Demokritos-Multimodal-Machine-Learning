import pandas as pd
import yaml
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from get_audio_features import get_audio_features
from get_video_features import get_video_features
from get_text_features import get_text_features, lyrics_to_embeddings


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


def get_features(test=False):

    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    FEATS_PATH = config["FEATS_PATH"]

    if not test:
        BLUES_PATH = config["BLUES_PATH"]
        COUNTRY_PATH = config["COUNTRY_PATH"]
        HIP_HOP_PATH = config["HIP_HOP_PATH"]
        INDIE_PATH = config["INDIE_PATH"]
        METAL_PATH = config["METAL_PATH"]
        POP_ROCK_PATH = config["POP_ROCK_PATH"]
        PUNK_PATH = config["PUNK_PATH"]
        SOUL_PATH = config["SOUL_PATH"]
        DRILL_PATH = config["DRILL_PATH"]


        clip_config = [BLUES_PATH, COUNTRY_PATH, HIP_HOP_PATH, INDIE_PATH, METAL_PATH, POP_ROCK_PATH, PUNK_PATH, SOUL_PATH, DRILL_PATH]

        index = create_index(config=clip_config)

        index.to_csv("./index.csv", index=False)

    else:

        TEST_PATH = config["TEST_PATH"]

        clip_config = [TEST_PATH]

        index = create_index(config=clip_config)

        index.to_csv("./index_test.csv", index=False)


    # print(index)

    audio_feats = get_audio_features(index[["path_audio", "song_name", "genre"]], config, test)
    #print(audio_feats)

    video_feats = get_video_features(index[["path_video", "song_name", "genre"]], config, test)
    #print(video_feats)

    get_text_features(index[["path_audio", "song_name", "genre"]], test)

    if not test:
        lyrics_to_embeddings(FEATS_PATH+"\\lyrics.csv", test)
    else:
        lyrics_to_embeddings(FEATS_PATH + "\\lyrics_test.csv", test)

# get_features()
