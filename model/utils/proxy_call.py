import requests
import json
import logging
import os

from openai import OpenAI

class OpenaiCall:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, messages, model="gpt-3.5-turbo-1106", temperature=0):
        response = self.client.chat.completions.create(
            model=model,
            # response_format={"type": "json_object"},
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content

    def stream_chat(self, messages, model="gpt-3.5-turbo-1106", temperature=0):
        for chunk in self.client.chat.completions.create(
            model=model,
            # response_format={"type": "json_object"},
            messages=messages,
            temperature=temperature,
            stream=True
        ):
            yield chunk.choices[0].delta.content
    
    def embedding(self, input_data):
        response = self.client.embeddings.create(
            input=input_data,
            model="text-embedding-3-small"
        )

        return response