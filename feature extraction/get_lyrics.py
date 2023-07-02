from ytmusicapi import YTMusic
from youtube_transcript_api import YouTubeTranscriptApi


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


yt = YTMusic("C:\\Users\\tatbo\\oauth.json")

# try and get lyrics from video-clip
video_title = "Airbourne - Back In The Game [OFFICIAL VIDEO].f140"
video_title = "Avenged Sevenfold - Hail To The King [Official Music Video].f140"
video_title = video_title[:-5]
print(video_title)

search_results = yt.search(video_title)

for i in range(len(search_results)):
    try:
        video_id = search_results[i]['videoId']
        print(video_id)
    except:
        continue

    transcript = get_transcript_from_yt_id(video_id)
    if transcript != "Transcript not found.":    # if lyrics are found, break
        print(f'{search_results[i]["videoId"]} : {transcript}')
        break

