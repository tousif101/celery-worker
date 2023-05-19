import os
import tempfile
import concurrent.futures
from celery import Celery
from supabase import create_client, Client
import openai
import uuid
from dotenv import load_dotenv
import time
from datetime import datetime
import tiktoken


load_dotenv()

# Load environment variables
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_API_KEY = os.environ['SUPABASE_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
REDIS_URL = os.environ['REDIS_BROKER']

openai.api_key = OPENAI_API_KEY
supabase: Client = create_client(SUPABASE_URL, SUPABASE_API_KEY)

app = Celery('tasks', broker=REDIS_URL)

def get_chat_gpt_response(prompt, transcript):
    model_name = 'text-davinci-003'
    max_tokens = 4080  # Change this to the model's actual maximum
    full_text = prompt + transcript
    encoding = tiktoken.get_encoding("p50k_base")
    num_tokens = len(encoding.encode(full_text))
    remaining_tokens = max_tokens - num_tokens

    # If remaining_tokens is negative, it means we are over the max_tokens limit
    if remaining_tokens < 0:
        print("Error: prompt and transcript length exceed max tokens.")
        return None

    # Generate the completion using the remaining tokens
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=remaining_tokens,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

@app.task
def process_video_and_save_transcript(user_id, transcript_id, fileSource):
    print(f"Starting Transcription for : {transcript_id}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{str(uuid.uuid4())}.mp3"
    file_path = os.path.join(tempfile.gettempdir(), filename)

    try:
        # Save the file from Supabase to the temporary file
        with open(file_path, 'wb+') as f:
            res = supabase.storage.from_('uploads').download(fileSource)
            # print(f"Downloaded file contents: {len(res)}")
            f.write(res)

        # Check the contents of the temporary file after it has been downloaded
        # with open(file_path, 'rb') as f:
        #     print(f"Temporary file contents: {f.read()}")

        # Transcribe video using Whisper ASR API
        with open(file_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)

        if transcript:
            transcript_text = transcript["text"]
            print(f"Transcript text: {transcript_text}")

            # Update the transcript and set status to 'completed' in Supabase
            table = supabase.table('transcripts')
            update_data = {
                'content': transcript_text,
                'status': 'completed'
            }
            response = table.update(update_data).eq('id', transcript_id).eq('user_id', user_id).execute()
            print(f"Transcript update response: {response}")

                        # Generate the analysis using ChatGPT
            prompts = [
                "Analyze the sales call transcript and assess the sales rep's product knowledge, including their ability to convey key features and benefits, and handle product-related objections. Provide a rating on a scale of 1-10 and offer suggestions for improvement.",
                "Identify areas where the sales rep could better tailor the product presentation to the customer's needs and preferences. Provide specific examples and recommendations.",
                "Analyze the sales call transcript and determine if the sales rep effectively positioned the product against competitors or alternative solutions. Provide a rating on a scale of 1-10 and suggest ways to improve the product positioning.",
                "Extract MEDDIC data from the sales call transcript and present it in a structured format.",
                "Analyze the sales call transcript and extract the following information: 1. Highlights of the call. 2. Ending statements from both the sales rep and the customer. 3. Details about any future call setups, including scheduled dates and times. 4. Positive and negative sentiments expressed during the call. 5. An assessment of how likely the buyer is to schedule another call. 6. Improvements and tips for the sales rep to enhance their performance. 7. Any discussion related to money or critical data points that can impact the deal."]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                analyses = list(executor.map(get_chat_gpt_response, prompts, [transcript_text]*len(prompts)))
            
            # Insert the analysis into the database
            analysis_data = {
                'transcript_id': transcript_id,
                'user_id': user_id,
                'product_knowledge_analysis': analyses[0],
                'tailoring_analysis': analyses[1],
                'product_positioning_analysis': analyses[2],
                'meddic_analysis': analyses[3],
                'additional_analysis': analyses[4],
            }
            response = supabase.table('analyses').insert(analysis_data).execute()
            print(f"Analysis insert response: {response}")

        else:
            print(f"Error transcribing video for transcript_id: {transcript_id}")

    finally:
        # Remove the temporary file after processing
        os.remove(file_path)
