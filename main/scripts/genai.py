import os
import openai
import json
import pandas as pd
import base64
import requests
import time
import cv2
import PyPDF2
from docx import Document
import re
import openai
from IPython.display import display, Image, HTML, Audio



class GenAI:
    def __init__(self, openai_api_key):
        self.client = openai.Client(api_key=openai_api_key)
        self.openai_api_key = openai_api_key

    def generate_text(self, prompt, instructions='You are helpul AI named Jarvis', model="gpt-4o-mini",
                       output_type = 'text'):
      '''Get a text completion from the OpenAI API'''
      completion = self.client.chat.completions.create(
                    model=model,
                    response_format={ "type": output_type},
                    messages=[
                      {"role": "system", "content": instructions},
                      {"role": "user", "content": prompt}
                    ]
                  )
      response =completion.choices[0].message.content

      return response

    def generate_chat_response(self, chat_history, user_message, instructions, model="gpt-4o-mini", output_type='text'):
        """
        Generate a chatbot response using the OpenAI API.

        Args:
            chat_history (list): The list of messages in the chat so far.
                                Each message is a dict with "role" and "content".
            user_message (str): The latest user message.
            instructions (str): The "system" instructions defining the chatbot's behavior.
            model (str): The OpenAI model to use. Default is "gpt-4o-mini".
            output_type (str): The format of the output. Default is 'text'.

        Returns:
            str: The chatbot's response.
        """
        # Add the latest user message to the chat history
        chat_history.append({"role": "user", "content": user_message})

        # Call the OpenAI API to get a response
        completion = self.client.chat.completions.create(
            model=model,
            response_format={"type": output_type},
            messages=[
                {"role": "system", "content": instructions},  # Add system instructions
                *chat_history  # Unpack the chat history to include all previous messages
            ]
        )

        # Extract the bot's response from the API completion
        bot_response = completion.choices[0].message.content

        # Add the bot's response to the chat history
        chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response


    def generate_image(prompt, client, model = "dall-e-3"):
      '''Generates an image using the OpenAI API'''

      response_img = self.client.images.generate(
        model= model,
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
      )
      time.sleep(1)
      image_url = response_img.data[0].url
      revised_prompt = response_img.data[0].revised_prompt

      return image_url, revised_prompt

    def encode_image(self,image_path):
      '''Encodes an image to base64'''
      with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_image_description(self, image_paths, instructions, model = 'gpt-4o-mini'):
        #print(f"instructions: {instructions}")
        '''Generates a description of a list of image_urls using the OpenAI Vision API'''
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        image_urls = [f"data:image/jpeg;base64,{self.encode_image(image_path)}" for image_path in image_paths]

        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instructions},
                            *map(lambda x: {"type": "image_url", "image_url": {"url": x}}, image_urls),
                            ],
            },
        ]
        params = {
            "model": model,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1000,
        }

        response = self.client.chat.completions.create(**params)
        image_description = response.choices[0].message.content
        return image_description

    def extract_frames(self, fname_video):
        '''Extract frames from a video file.'''
        if not os.path.exists(fname_video):
            
            return [], 0, 0

        video = cv2.VideoCapture(fname_video)  # open the video file
        if not video.isOpened():
            #logger.error(f"Failed to open video file: {fname_video}")
            return [], 0, 0

        nframes = video.get(cv2.CAP_PROP_FRAME_COUNT)  # number of frames in video
        fps = video.get(cv2.CAP_PROP_FPS)  # frames per second in video

        #logger.debug(f"{nframes} frames in video")
        #logger.debug(f"{fps} frames per second")

        base64Frames = []
        max_samples = 15
        frame_interval = max(1, int(nframes // max_samples))  # Calculate the interval at which to sample frames

        current_frame = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            if current_frame % frame_interval == 0 and len(base64Frames) < max_samples:
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            current_frame += 1

        video.release()

        return base64Frames, nframes, fps

    def generate_video_description(self,fname_video, instructions, model = 'gpt-4o-mini'):
        '''Get narration from a video file'''
        #print('Sample video frames ...')
        base64Frames_samples, nframes, fps = self.extract_frames(fname_video)
        wps = 200 / 60  # words per second in normal speech
        nwords_max = round(nframes / fps * wps)  # max number of words in the voice over
        image_urls = [f"data:image/jpeg;base64,{base64_image}" for base64_image in base64Frames_samples]
        
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instructions},
                            *map(lambda x: {"type": "image_url", "image_url": {"url": x}}, image_urls),
                            ],
            },
        ]
        params = {
            "model": model,
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1000,
        }

        response = self.client.chat.completions.create(**params)
        image_description = response.choices[0].message.content

        return image_description




    def recognize_speech(self,audio_filename):
        try:
            
            #print("Load audio file")
            audio_file= open(audio_filename, "rb")

            #print("\ttranscribe audio")
            transcription = self.client.audio.transcriptions.create(
              model="whisper-1", 
              file=audio_file
            )
            # Print the transcribed text
            #print(transcription.text)
            
            return transcription.text
        except Exception as e:
            
            traceback.print_exc()
            return None


    def read_pdf(self,file_path):
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Initialize the PDF reader
            reader = PyPDF2.PdfReader(file)
            
            # Initialize an empty string to store the text
            text = ""
            
            # Iterate through each page in the PDF
            for page in reader.pages:
                # Extract the text from the page and add it to the text string
                text += page.extract_text()
            
        return text



    def read_docx(self,file_path):
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)


    def get_embedding(self, text, model = 'text-embedding-3-small'):
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(
                    input=text,
                    model=model
                )
        return response.data[0].embedding

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def display_tweet(self,text='life is good', screen_name='zlisto'):
        display_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .tweet {{
                    background-color: white;
                    color: black;
                    border: 1px solid #e1e8ed;
                    border-radius: 10px;
                    padding: 20px;
                    max-width: 500px;
                    margin: 20px auto;
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                }}
                .user strong {{
                    color: #1da1f2;
                }}
                .tweet-text p {{
                    margin: 0;
                    line-height: 1.5;
                }}
            </style>
        </head>
        <body>
            <div class="tweet">
                <div class="user">
                    <strong>@{screen_name}</strong>
                </div>
                <div class="tweet-text">
                    <p>{text}</p>
                </div>
            </div>
        </body>
        </html>
        '''
        display(HTML(display_html))
        return display_html

    def display_IG(self,caption, image_url, screen_name=None, profile_image_url = None):
        ''' HTML template for displaying the image, screen name, and caption in an Instagram-like format'''

        display_html = f"""
        <style>
            .instagram-post {{
                border: 1px solid #e1e1e1;
                border-radius: 3px;
                width: 600px;
                margin: 20px auto;
                background-color: white;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            }}
            .instagram-header {{
                padding: 14px;
                border-bottom: 1px solid #e1e1e1;
                display: flex;
                align-items: center;
            }}
            .instagram-profile-pic {{
                border-radius: 50%;
                width: 32px;
                height: 32px;
                margin-right: 10px;
            }}
            .instagram-screen-name {{
                font-weight: bold;
                color: #262626;
                text-decoration: none;
                font-size: 14px;
            }}
            .instagram-image {{
                max-width: 600px;
                width: auto;
                height: auto;
                display: block;
                margin: auto;
            }}
            .instagram-caption {{
                padding: 10px;
                font-size: 14px;
                color: #262626;
            }}
            .instagram-footer {{
                padding: 10px;
                border-top: 1px solid #e1e1e1;
            }}
            .instagram-likes {{
                font-weight: bold;
                margin-bottom: 8px;
            }}
        </style>
        <div class="instagram-post">
            <div class="instagram-header">
                <img src="{profile_image_url}" alt="Profile picture" class="instagram-profile-pic">
                <a href="#" class="instagram-screen-name">{screen_name}</a>
            </div>
            <img src="{image_url}" alt="Instagram image" class="instagram-image">
            <div class="instagram-caption">
                <a href="#" class="instagram-screen-name">{screen_name}</a> {caption}
            </div>
            <div class="instagram-footer">
                <div class="instagram-likes">24 likes</div>
                <!-- Include other footer content here -->
            </div>
        </div>
        """
        display(HTML(display_html))
        return display_html