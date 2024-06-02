import openai
from IPython.display import display, Image, HTML, Audio
import base64
import requests
import time 
def generate_text(prompt, instructions, client, model="gpt-4o",
                   output_type = 'text'):
  '''Get a text completion from the OpenAI API'''
  completion = client.chat.completions.create(
                model=model,
                response_format={ "type": output_type},
                messages=[
                  {"role": "system", "content": instructions},
                  {"role": "user", "content": prompt}
                ]
              )
  response =completion.choices[0].message.content

  return response

def generate_image(prompt, client, model = "dall-e-3"):
  '''Generates an image using the OpenAI API'''

  response_img = client.images.generate(
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

def generate_image_description(image_urls, instructions, client):
  '''Generates a description of a list of image_urls using the OpenAI Vision API'''
  PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [{"type": "text","text":instructions},
            *map(lambda x: {"type":"image_url","image_url": {"url":x}}, image_urls),
        ],
    },
  ]
  params = {
      "model": "gpt-4o",
      "messages": PROMPT_MESSAGES,
      "max_tokens": 1000,
  }

  response= client.chat.completions.create(**params)


  image_description = response.choices[0].message.content
  return image_description

def encode_image(image_path):
  '''Encodes an image to base64'''
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def display_image_url(image_url, width = 500, height = 500):
  '''Create static url for image located at image_url so it remains in the notebook
  even after the link dies '''
  response = requests.get(image_url)
  image_data = response.content
  # Encoding the image data as base64
  base64_image = base64.b64encode(image_data).decode('utf-8')
  # Generating HTML to display the image
  html_code = f'<img src="data:image/jpeg;base64,{base64_image}" width="{width}" height="{height}"/>'
  
  return html_code


def display_tweet(text='life is good', screen_name='zlisto'):
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

def display_IG(caption, image_url, screen_name=None, profile_image_url = None):
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

