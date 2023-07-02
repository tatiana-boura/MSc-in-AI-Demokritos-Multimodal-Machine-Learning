from ytmusicapi import YTMusic
from youtube_transcript_api import YouTubeTranscriptApi

import torch
import pandas as pd
from transformers import BertTokenizer, BertModel

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


def get_text_features(data, test=False):

    text_feats_df = pd.DataFrame(columns=["song_name", "lyrics"])
    yt = YTMusic("C:\\Users\\tatbo\\oauth.json")

    no_lyrics = 0

    for i in range(len(data)):
    #for i in range(10):

        video_title = data.loc[i].song_name

        search_results = yt.search(video_title)

        transcript = None
        for j in range(len(search_results)):
            try:
                video_id = search_results[j]['videoId']
            except:
                continue

            transcript = get_transcript_from_yt_id(video_id)
            if transcript != "Transcript not found." and len(transcript) > 10:  # if lyrics are found, break

                transcript = transcript.replace("\n", "")
                transcript = transcript.replace("[Music]", "")
                transcript = transcript.replace("♪", "")

                # print(f'{search_results[j]["videoId"]} : {transcript}')
                text_feats_df.loc[len(text_feats_df)] = [video_title, transcript]
                break

        if transcript == "Transcript not found.":
            # text_feats_df.loc[len(text_feats_df)] = [video_title, " "]
            print(f'{video_title} : no lyrics found :(')
            no_lyrics += 1

    print(no_lyrics)
    if not test:
        text_feats_df.to_csv("./lyrics.csv", index=False)
    else:
        text_feats_df.to_csv("./lyrics_test.csv", index=False)


def lyrics_to_embeddings(df_path, test=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device is: {device}')

    df = pd.read_csv(df_path)
    text_feats_df = pd.DataFrame(columns=["song_name"]+[str(i) for i in range(768)])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)

    for i in range(len(df)):

        if (i + 1) % 100 == 0:
            print(f'Lyrics of video-clip {i}...')

        lyrics = df.loc[i].lyrics

        tokens = tokenizer.encode(lyrics, add_special_tokens=True)
        if len(tokens) > 512:
            tokens = tokens[:512]
        inputs = torch.tensor(tokens).unsqueeze(0)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        embedding = outputs.last_hidden_state[:, 0, :].to("cpu").tolist()

        text_feats_df.loc[len(text_feats_df)] = [df.loc[i].song_name]+embedding[0]

        if not test:
            text_feats_df.to_csv("./lyrics_embeddings.csv", index=False)
        else:
            text_feats_df.to_csv("./lyrics_embeddings_test.csv", index=False)