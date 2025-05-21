import openai
import sympy as sp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from transformers import pipeline
import requests
import logging
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
import configparser
import speech_recognition as sr
import pyttsx3
import sqlite3
from googletrans import Translator
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
from textblob import TextBlob
import schedule
import time
from bs4 import BeautifulSoup

# Load configuration
config = configparser.ConfigParser()
config.read("config.ini")

# Set up OpenAI API key
openai.api_key = config["API_KEYS"]["openai_api_key"]

# Initialize Hugging Face summarization pipeline
summarizer = pipeline("summarization")

# Initialize Wolfram Alpha API
WOLFRAM_APP_ID = config["API_KEYS"]["wolfram_app_id"]

# Set up logging
logging.basicConfig(filename="tofetai.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Initialize Rich console for stylish output
console = Console()

# Initialize speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Initialize translator
translator = Translator()

# Initialize SQLite database for persistent memory
conn = sqlite3.connect("tofetai.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memory (input TEXT, response TEXT, timestamp TEXT)")
conn.commit()

class TofetAI:
    def __init__(self):
        self.console = console
        self.personality = "friendly"  # Default personality
        self.user_preferences = {}  # Store user preferences

    def log_interaction(self, user_input, response):
        """Log user interactions to the database."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO memory (input, response, timestamp) VALUES (?, ?, ?)", (user_input, response, timestamp))
        conn.commit()
        logging.info(f"User: {user_input}\nTofetAI: {response}")

    async def natural_language_processing(self, text):
        """Basic NLP using OpenAI's GPT model."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Thinking...", total=1)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=text,
                max_tokens=150,
                temperature=0.7 if self.personality == "casual" else 0.3
            )
            progress.update(task, completed=1)
        return response.choices[0].text.strip()

    async def summarize_text(self, text):
        """Summarize text using Hugging Face's summarization pipeline."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Summarizing...", total=1)
            summary = summarizer(text, max_length=50, min_length=25, do_sample=False)
            progress.update(task, completed=1)
        return summary[0]['summary_text']

    async def solve_math_equation(self, equation):
        """Solve math equations using SymPy."""
        try:
            solution = sp.solve(equation)
            return f"Solution: {solution}"
        except Exception as e:
            return f"Error solving equation: {e}"

    async def predict_future_trends(self, data):
        """Predict future trends using linear regression."""
        X = np.array(range(len(data))).reshape(-1, 1)
        y = np.array(data)
        model = LinearRegression()
        model.fit(X, y)
        future = model.predict([[len(data)]])
        return f"Predicted next value: {future[0]}"

    async def query_wolfram_alpha(self, query):
    """Query Wolfram Alpha for complex science and math problems."""
    url = f"http://api.wolframalpha.com/v2/query?input={query}&appid={WOLFRAM_APP_ID}"
    with Progress() as progress:
        task = progress.add_task("[cyan]Querying Wolfram Alpha...", total=1)
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            progress.update(task, completed=1)
            return response.text
        except requests.exceptions.RequestException as e:
            progress.update(task, completed=1)
            return f"Error querying Wolfram Alpha: {e}"
        
    async def generate_code(self, task):
        """Generate code using OpenAI's Codex model."""
        with Progress() as progress:
            task_progress = progress.add_task("[cyan]Generating code...", total=1)
            response = openai.Completion.create(
                engine="code-davinci-002",
                prompt=task,
                max_tokens=200
            )
            progress.update(task_progress, completed=1)
        return response.choices[0].text.strip()

    async def entertain_user(self):
        """Cure boredom with a joke or fun fact."""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return np.random.choice(jokes)

    async def automate_task(self, task):
        """Simulate task automation."""
        return f"Task '{task}' has been automated and completed."

    async def interact_with_data(self, data, query):
        """Interact with data using Pandas."""
        df = pd.DataFrame(data)
        try:
            result = df.query(query)
            return result
        except Exception as e:
            return f"Error querying data: {e}"

    async def translate_text(self, text, target_language="en"):
        """Translate text to a target language."""
        translation = translator.translate(text, dest=target_language)
        return translation.text

    async def voice_input(self):
        """Capture voice input from the user."""
        with sr.Microphone() as source:
            self.console.print("[cyan]Listening...[/]")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                self.console.print(f"[green]You: {text}[/]")
                return text
            except sr.UnknownValueError:
                return "Sorry, I could not understand that."

    async def voice_output(self, text):
        """Convert text to speech."""
        engine.say(text)
        engine.runAndWait()

    async def set_personality(self, personality):
        """Set the chatbot's personality."""
        self.personality = personality
        return f"Personality set to {personality}."

    async def get_gnews_updates(self):
        """Fetch and summarize the latest news using GNews API."""
        api_key = config["API_KEYS"]["gnews_api_key"]
        url = f"https://gnews.io/api/v4/top-headlines?token={api_key}&lang=en"
        with Progress() as progress:
            task = progress.add_task("[cyan]Fetching news...", total=1)
            response = requests.get(url)
            progress.update(task, completed=1)
        articles = response.json().get("articles", [])
        news_summary = "\n".join([f"{article['title']}" for article in articles[:5]])
        return f"Here are the top news headlines:\n{news_summary}"

    async def play_game(self):
        """Play a simple trivia game."""
        questions = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is the largest planet in the solar system?", "answer": "Jupiter"}
        ]
        question = np.random.choice(questions)
        self.console.print(Panel.fit(Text(question["question"], style="bold green")))
        user_answer = await self.voice_input()
        if user_answer.lower() == question["answer"].lower():
            return "Correct! Well done!"
        else:
            return f"Wrong! The correct answer is {question['answer']}."

    async def analyze_sentiment(self, text):
        """Analyze the sentiment of user input."""
        analysis = TextBlob(text)
        sentiment = "positive" if analysis.sentiment.polarity > 0 else "negative" if analysis.sentiment.polarity < 0 else "neutral"
        return f"The sentiment of your input is {sentiment}."

    async def visualize_data(self, data):
        """Visualize data using matplotlib."""
        df = pd.DataFrame(data)
        try:
            df.plot(kind="bar")
            plt.show()
            return "Here's the visualization of your data!"
        except Exception as e:
            return f"Error visualizing data: {e}"

    async def generate_art(self, prompt):
        """Generate art using OpenAI's DALL·E."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating art...", total=1)
            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="256x256"
            )
            progress.update(task, completed=1)
        image_url = response['data'][0]['url']
        return f"Here's your generated art: {image_url}"

    async def set_reminder(self, reminder, time):
        """Set a reminder for a specific time."""
        schedule.every().day.at(time).do(lambda: self.voice_output(f"Reminder: {reminder}"))
        return f"Reminder set for {time}."

    async def learn_language(self):
        """Help users learn a new language."""
        return "Let's learn a new language! What language are you interested in?"

    async def file_operation(self, operation, filename, content=None):
        """Perform file operations."""
        if operation == "create":
            with open(filename, "w") as file:
                file.write(content)
            return f"File '{filename}' created."
        elif operation == "read":
            with open(filename, "r") as file:
                return file.read()
        elif operation == "delete":
            os.remove(filename)
            return f"File '{filename}' deleted."
        else:
            return "Invalid file operation."

    async def scrape_website(self, url):
        """Scrape data from a website."""
        with Progress() as progress:
            task = progress.add_task("[cyan]Scraping website...", total=1)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, "html.parser")
            progress.update(task, completed=1)
        return f"Scraped data: {soup.title.text}"

    async def detect_emotion(self, text):
        """Detect emotion from text."""
        # Placeholder for emotion detection logic
        return "Emotion detection is under development."

    async def control_device(self, device, action):
        """Control smart home devices."""
        # Placeholder for smart home integration
        return f"Device '{device}' {action}ed."

    async def recommend_resources(self, interest):
        """Provide personalized learning resources."""
        # Placeholder for recommendation logic
        return f"Here are some resources for {interest}: [Resource 1, Resource 2]"

    async def run(self):
        """Main loop to interact with the user."""
        self.console.print(Panel.fit(Text("Welcome to TofetAI - Assistant!", style="bold blue")))
        self.console.print("How can I help you today? Type 'exit' to quit.")

        while True:
            user_input = await self.voice_input()  # Use voice input
            if user_input.lower() in ["exit", "quit"]:
                self.console.print(Panel.fit(Text("Goodbye! Have a great day!", style="bold red")))
                break

            response = await self.handle_task(user_input)
            self.console.print(Panel.fit(Text(f"TofetAI: {response}", style="bold magenta")))
            await self.voice_output(response)  # Use voice output
            self.log_interaction(user_input, response)  # Log interaction

# Run the chatbot
if __name__ == "__main__":
    chatbot = TofetAI()
    asyncio.run(chatbot.run())