from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from openai import OpenAI
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup


load_dotenv(override=True)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_video_title(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    return soup.title.string.replace(" - YouTube", "").strip()

def get_video_id(url):
    parts = url.split('?v=')
    if len(parts) > 1:
        video_id = parts[1]
    return video_id

def get_youtube_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.get_transcript(video_id)
        # Join all text segments
        full_text = ' '.join([entry['text'] for entry in transcript])

        return full_text
    except (TranscriptsDisabled, NoTranscriptFound):
        return "Transcript not available for this video."
    
class VideoChatbot:
    def __init__(self, url):
        self.url = url
        self.title = get_video_title(url)
        self.video_id = get_video_id(url)
        self.transcript = get_youtube_transcript(self.video_id)
        self.history = []
        self.system_prompt = {
            "role": "system",
            "content": f"You are an expert assistant that analyzes the transcript of youtube video and who only answers questions using the transcript of the YouTube video '{self.title}'. When relevant, answer with specific details from the transcript."
        }
    
    def summarize(self):
        prompt = f"Summarize the following YouTube video transcript in detail:\n\n{self.transcript}\n\n Answer in Markdown."
        messages = [self.system_prompt, {"role": "user", "content": prompt}]
        resp = openai.chat.completions.create(
            model="gpt-4o-mini", messages=messages
        )
        summary = resp.choices[0].message.content
        self.history = [self.system_prompt, {"role": "user", "content": prompt}, {"role": "assistant", "content": summary}]
        return summary

    def chat(self, user_input):
        self.history.append({"role": "user", "content": user_input})
        resp = openai.chat.completions.create(
            model="gpt-4o-mini", messages=self.history
        )
        answer = resp.choices[0].message.content
        self.history.append({"role": "assistant", "content": answer})
        return answer
        


if __name__ == "__main__":
    url = input("Paste your YouTube URL: ")
    bot = VideoChatbot(url)
    print("----- SUMMARY -----")
    print(bot.summarize())
    print("\nNow, ask questions about the video!")
    while True:
        q = input("\nQ: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("\nA:", bot.chat(q))