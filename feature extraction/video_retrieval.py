from ytmusicapi import YTMusic
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp as youtube_dl
import os
import yaml
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from get_features import get_features
import subprocess

import sys

# adding Folder_2 to the system path
sys.path.insert(0, '../')

from autoencoder import get_representations


def get_transcript_from_yt_id(video_id):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ""

        for segment in transcript_list:
            text = segment['text']
            transcript += text + " "

        return transcript.strip()
    except:
        return "Transcript not found."


def download_video(video_title, store_path):

    # Set up the youtube-dl options
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'{store_path}%(title)s.%(ext)s',  # Specify the output file template
        'ignoreerrors': True  # Ignore extraction errors
    }

    text_feats_df = pd.DataFrame(columns=["song_name", "lyrics"])
    yt = YTMusic("C:\\Users\\tatbo\\oauth.json")

    search_results = yt.search(video_title)

    downloaded = False

    for j in range(len(search_results)):
        try:
            video_id = search_results[j]['videoId']

            if downloaded == False:  # if best match, download video
                # Download video
                video_url = f'https://www.youtube.com/watch?v={video_id}'

                with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                    try:
                        ydl.download([video_url])
                        downloaded = True

                    except youtube_dl.utils.DownloadError as e:
                        if 'Private video' in str(e):
                            print(f'Skipped private video: {video_url}')
                        else:
                            raise
        except:
            continue



def split_mp4(dir_path):

    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            video_path = dir_path+path

    audio_output = video_path[:-4]+".m4a"
    video_output = video_path[:-4]+"1.mp4"

    command = ['ffmpeg', '-i', video_path, '-c:v', 'copy', '-an', video_output, '-vn', '-c:a', 'copy', audio_output]
    subprocess.run(command)


def retrieve_closest_videos(video_title):
    """
    :param video_title: Name of video clip in YouTube
    :return: TBD
    """

    # First, download the video
    with open('../config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    TEST_PATH = config["TEST_PATH"]
    EMBEDDINGS_PATH = config["EMBEDDINGS_PATH"]
    EMBEDDINGS_TEST_PATH = config["EMBEDDINGS_TEST_PATH"]


    download_video(video_title, TEST_PATH)

    split_mp4(TEST_PATH)

    get_features(test=True)

    get_representations()

    # Now get 5 closest

    embeddings_train = pd.read_csv(EMBEDDINGS_PATH)
    embeddings_test = pd.read_csv(EMBEDDINGS_TEST_PATH)

    print(embeddings_test.shape)
    print(embeddings_train.shape)
    for i in range(embeddings_test.shape[0]):
        query = embeddings_test.iloc[i, 1:]

        # Calculate cosine similarity
        cos_sim = cosine_similarity(embeddings_train.iloc[:, 1:].values, query.values.reshape(1, -1))

        # Create a dataframe with cosine similarity values and row indices
        similarity_df = pd.DataFrame({'similarity': cos_sim.flatten(), 'index': embeddings_train.index})

        # Sort dataframe by similarity in descending order
        sorted_df = similarity_df.sort_values(by='similarity', ascending=False)

        # Get the top k rows
        k = 5  # Replace with your desired value of k
        top_k_rows = embeddings_train.loc[sorted_df['index'].head(k)]

        print(f'Video clip: {embeddings_test.iloc[i, 0]} similar to the following')
        print(top_k_rows.song_name)



