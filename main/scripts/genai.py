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
    """
    A class for interacting with the OpenAI API to generate text, images, video descriptions,
    perform speech recognition, and handle basic document processing tasks.

    Attributes:
    ----------
    client : openai.Client
        An instance of the OpenAI client initialized with the API key.
    """
    def __init__(self, openai_api_key):
        """
        Initializes the GenAI class with the provided OpenAI API key.

        Parameters:
        ----------
        openai_api_key : str
            The API key for accessing OpenAI's services.
        """
        self.client = openai.Client(api_key=openai_api_key)
        self.openai_api_key = openai_api_key

    def generate_text(self, prompt, instructions='You are a helpful AI named Jarvis', model="gpt-4o-mini", output_type='text', temperature =1):
        """
        Generates a text completion using the OpenAI API.

        This function sends a prompt to the OpenAI API with optional instructions to guide the AI's behavior. 
        It supports specifying the model and output format, and returns the generated text response.

        Parameters:
        ----------
        prompt : str
            The user input or query that you want the AI to respond to.
        
        instructions : str, optional (default='You are a helpful AI named Jarvis')
            System-level instructions to define the AI's behavior, tone, or style in the response.
        
        model : str, optional (default='gpt-4o-mini')
            The OpenAI model to use for generating the response. You can specify different models like 'gpt-4', 'gpt-3.5-turbo', etc.
        
        output_type : str, optional (default='text')
            The format of the output. Typically 'text', but can be customized for models that support different response formats.

        Returns:
        -------
        str
            The AI-generated response as a string based on the provided prompt and instructions.

        Example:
        -------
        >>> response = generate_text("What's the weather like today?")
        >>> print(response)
        "The weather today is sunny with a high of 75°F."
        """
        completion = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": output_type},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        response = response.replace("```html", "")
        response = response.replace("```", "")
        return response


    def generate_chat_response(self, chat_history, user_message, instructions, model="gpt-4o-mini", output_type='text'):
        """
        Generates a chatbot-like response based on the conversation history.

        Parameters:
        ----------
        chat_history : list
            List of previous messages, each as a dict with "role" and "content".
        user_message : str
            The latest message from the user.
        instructions : str
            System instructions defining the chatbot's behavior.
        model : str, optional
            The OpenAI model to use (default is 'gpt-4o-mini').
        output_type : str, optional
            The format of the output (default is 'text').

        Returns:
        -------
        str
            The chatbot's response.
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


    def generate_image(self, prompt, model="dall-e-3", size="1024x1024", quality="standard", n=1):
        """
        Generates an image from a text prompt using the OpenAI DALL-E API.

        Parameters:
        ----------
        prompt : str
            The description of the image to generate. This text guides the AI to create an image
            based on the provided details.
        model : str, optional
            The OpenAI model to use for image generation. Defaults to 'dall-e-3'.
        size : str, optional
            The desired dimensions of the generated image. Defaults to '1024x1024'.
            Supported sizes may vary depending on the model.
        quality : str, optional
            The quality of the generated image, such as 'standard' or 'high'. Defaults to 'standard'.
        n : int, optional
            The number of images to generate. Defaults to 1.

        Returns:
        -------
        tuple
            A tuple containing:
            - image_url (str): The URL of the generated image.
            - revised_prompt (str): The prompt as modified by the model, if applicable.

        Notes:
        -----
        This function introduces a short delay (`time.sleep(1)`) to ensure proper API response handling.
        """
        response_img = self.client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        time.sleep(1)
        image_url = response_img.data[0].url
        revised_prompt = response_img.data[0].revised_prompt

        return image_url, revised_prompt

    def display_image_url(self,image_url, width=256, height=256):
        """
        Creates a static, embeddable HTML representation of an image from a given URL,
        ensuring the image remains viewable even if the original link becomes inactive.

        Parameters:
        ----------
        image_url : str
            The URL of the image to be displayed.
        width : int, optional
            The width (in pixels) to display the image. Defaults to 500.
        height : int, optional
            The height (in pixels) to display the image. Defaults to 500.

        Returns:
        -------
        str
            An HTML string containing the base64-encoded image, which can be embedded
            directly into a notebook or web page.

        Notes:
        -----
        - The function downloads the image from the provided URL and encodes it in base64,
        ensuring it remains static even if the original URL is no longer accessible.
        - This approach is useful for displaying images in environments like Jupyter Notebooks,
        where image persistence is desired.
        """# Validate that image_url is a proper string and has a valid URL scheme
        if not isinstance(image_url, str) or not image_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid image URL provided: {image_url}")

        response = requests.get(image_url)
        image_data = response.content
        # Encoding the image data as base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        # Generating HTML to display the image
        html_code = f'<img src="data:image/jpeg;base64,{base64_image}" width="{width}" height="{height}"/>'
        
        return html_code

    def encode_image(self,image_path):
        """
        Encodes an image file into a base64 string.

        Parameters:
        ----------
        image_path : str
            The path to the image file.

        Returns:
        -------
        str
            Base64-encoded image string.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_image_description(self, image_paths, instructions, model = 'gpt-4o-mini'):
        """
        Generates a description for one or more images using OpenAI's vision capabilities.

        Parameters:
        ----------
        image_paths : str or list
            Path(s) to the image file(s).
        instructions : str
            Instructions for the description.
        model : str, optional
            The OpenAI model to use (default is 'gpt-4o-mini').

        Returns:
        -------
        str
            A textual description of the image(s).
        """
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

        completion = self.client.chat.completions.create(**params)
        response = completion.choices[0].message.content
        response = response.replace("```html", "")
        response = response.replace("```", "")
        return response

    def extract_frames(self, fname_video, max_samples = 15):
        """
        Extracts frames from a video file at regular intervals.

        Parameters:
        ----------
        video_path : str
            Path to the video file.

        Returns:
        -------
        tuple
            A tuple containing:
            - A list of base64-encoded image frames
            - Total number of frames in the video
            - Frames per second (FPS) of the video
        """
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

    def generate_video_description(self, fname_video, instructions, max_samples=15, model='gpt-4o-mini'):
        """
        Generates a textual description of a video by analyzing sampled frames.

        Parameters
        ----------
        fname_video : str
            Path to the video file.
        instructions : str
            Guidelines for generating the description.
        max_samples : int, optional
            Maximum number of frames to sample from the video (default is 15).
        model : str, optional
            OpenAI model used for generating the description (default is 'gpt-4o-mini').

        Returns
        -------
        str
            A descriptive summary of the video content.
        """
        # Extract sampled frames and video metadata
        base64Frames_samples, nframes, fps = self.extract_frames(fname_video, max_samples)

        # Estimate the maximum number of words based on speech rate
        words_per_second = 200 / 60  # Typical speech rate
        max_words = round(nframes / fps * words_per_second)

        # Convert frames to base64 image URLs
        image_urls = [f"data:image/jpeg;base64,{base64_image}" for base64_image in base64Frames_samples]

        # Prepare API prompt messages
        prompt_messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": instructions}] +
                        [{"type": "image_url", "image_url": {"url": url}} for url in image_urls],
            },
        ]

        # API request parameters
        params = {
            "model": model,
            "messages": prompt_messages,
            "max_tokens": 1000,
        }

        # Generate completion using OpenAI's API
        completion = self.client.chat.completions.create(**params)
        response = completion.choices[0].message.content

        # Clean up response formatting
        return response.replace("```html", "").replace("```", "")

    def generate_audio(self, text, file_path, model='tts-1', voice='nova', speed=1.0):
        """
        Generates an audio file from the given text using OpenAI's text-to-speech (TTS) model.

        Parameters
        ----------
        text : str
            The input text to be converted into speech.
        file_path : str
            The output file path where the generated audio will be saved.
        model : str, optional
            The OpenAI TTS model to use (default is 'tts-1').
        voice : str, optional
            The voice to use for synthesis. Available voices:
            - 'nova' (default)
            - 'alloy'
            - 'echo'
            - 'fable'
            - 'onyx'
            - 'shimmer'
        speed : float, optional
            The speech speed multiplier (default is 1.0).

        Returns
        -------
        bool
            Returns True if the audio file is successfully generated and saved.
        """

        # Generate speech using OpenAI's API
        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=speed  # Include speed parameter
        )

        # Save the generated audio to the specified file path
        response.stream_to_file(file_path)

        return True





    def recognize_speech(self,audio_filename, model = 'whisper-1'):
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


    def get_embedding(self, text, model='text-embedding-3-small'):
        """
        Generates an embedding vector for a given text using the OpenAI embedding model.

        Parameters:
        ----------
        text : str
            The input text to be converted into an embedding. Newline characters (`\n`) 
            are replaced with spaces to ensure proper processing.
        model : str, optional
            The OpenAI embedding model to use. Defaults to 'text-embedding-3-small'.

        Returns:
        -------
        list
            A list of floating-point numbers representing the embedding vector of the input text.

        Notes:
        -----
        - Embeddings are useful for tasks such as semantic search, clustering, and classification.
        - The function replaces newline characters in the input text with spaces before processing.
        """
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