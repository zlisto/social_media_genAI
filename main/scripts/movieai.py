import os
import shutil
import json
import glob
import time
import ast
import tempfile
import subprocess
import pandas as pd
from tqdm.auto import tqdm  # Ensures compatibility in Jupyter and Colab
from scripts.genai import GenAI  # Import base class



class MovieAI(GenAI):
    """
    A subclass of GenAI specifically designed for movie-related AI tasks, 
    such as analyzing scenes, generating subtitles, and summarizing movies.
    """


    def __init__(self, openai_api_key, ffmpeg_path="ffmpeg.exe"):
        """
        Initializes MovieAI as an extension of GenAI.

        Parameters:
        ----------
        openai_api_key : str
            The API key for accessing OpenAI's services.
        ffmpeg_path : str, optional (default="ffmpeg.exe")
            The path to the FFmpeg executable, used for video processing.
        """

        super().__init__(openai_api_key)  # Initialize parent class (GenAI)
        self.ffmpeg_path = ffmpeg_path

        # Check if FFmpeg is accessible
        if not shutil.which(self.ffmpeg_path):
            raise FileNotFoundError(
                f"FFmpeg not found at '{self.ffmpeg_path}'. Please ensure FFmpeg is installed and available in PATH."
            )
    

    def split_video(self, file_path: str, output_directory: str, segment_time: int = 60) -> None:
        """
        Splits a video file into multiple clips of specified duration using FFmpeg.
        If the output directory exists, it clears all files before saving new clips.

        Parameters:
        ----------
        file_path : str
            Path to the input video file.
        output_directory : str
            Directory to save the output clips. If it exists, all existing files inside will be deleted.
        segment_time : int, optional
            Duration (in seconds) of each clip (default: 60 seconds).

        Returns:
        -------
        None
        """

        # Ensure input file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Error: The input file '{file_path}' does not exist.")

        # Delete all existing files in output directory (if it exists)
        if os.path.exists(output_directory):
            print(f"🗑️ Clearing existing files in '{output_directory}'...")
            for filename in os.listdir(output_directory):
                file_path = os.path.join(output_directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete file/symlink
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # Delete subdirectories
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")

        # Recreate the output directory
        os.makedirs(output_directory, exist_ok=True)

        # Define output file naming pattern
        output_pattern = os.path.join(output_directory, "clip_%03d.mp4")

        # FFmpeg command
        command = [
            self.ffmpeg_path,  # Use full path to ffmpeg executable
            "-i", file_path,
            "-c", "copy",  # Copy codec (fast processing)
            "-map", "0",
            "-segment_time", str(segment_time),
            "-f", "segment",
            "-reset_timestamps", "1",
            output_pattern
        ]

        # Run FFmpeg
        try:
            print(f"🎬 Splitting video into {segment_time}-second clips...")
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ Video successfully split into clips at '{output_directory}'.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: FFmpeg encountered an issue.\n{e.stderr.decode('utf-8')}")



    def generate_clip_descriptions(self, clip_paths, instructions_base="", model = 'gpt-4o-mini', verbose = False):
        """
        Generates a detailed description of each movie clip in `clip_paths`.

        Parameters:
        ----------
        clip_paths : list of str
            List of file paths for video clips to be analyzed.
        instructions_base : str, optional
            Additional context or instructions to be added to the prompt.
        model : str, optional
            LLM model to use for generating descriptions (default: 'gpt-4o-mini').
        verbose : bool, optional
            Whether to display the descriptions as they are generated (default: False).

        Returns:
        -------
        pd.DataFrame
            A DataFrame with columns ["clip_path", "description"] containing descriptions for each clip.
            If an error occurs for a clip, it is skipped.
            If no clips are successfully processed, returns `False`.
        """

        dict_list = []
        description = "This is the first clip, so no previous scene."

        for clip_path in tqdm(clip_paths, desc="Processing Clips", unit="clip"):
            try:
                instructions = f"""{instructions_base} Generate a detailed description of this clip from a longer video.
                                 The previous clip in the sequence had a description:{description}"""

                print(f"Processing: {clip_path}")

                # Generate description
                description = self.generate_video_description(
                    clip_path, 
                    instructions, 
                    max_samples=10, 
                    model=model
                )
                if verbose:
                    print(f"📝 Description for {clip_path}: {description}")

                dict_list.append({"clip_path": clip_path, "description": description})

            except Exception as e:
                print(f"❌ Error processing {clip_path}: {e}")
                continue  # Skip this clip and move to the next

        # Return DataFrame if at least one clip was processed, otherwise return False
        return pd.DataFrame(dict_list) if dict_list else False

                

    def generate_summary_script(self, df_clips, instructions, model='gpt-4o-mini'):
        """
        Generates a script for a summary video based on clip descriptions.

        Parameters:
        ----------
        df_clips : pd.DataFrame
            A DataFrame containing clip descriptions. It should have a "clip_path" column.
        instructions : str
            Additional guidance for selecting clips and structuring the summary video.
        model : str, optional (default='gpt-4o-mini')
            The OpenAI model used for text generation.

        Returns:
        -------
        pd.DataFrame or bool
            A DataFrame with columns ["clip_path", "narration"] containing the summary script.
            If an error occurs, returns `False`.
        """

        try:
            # Convert DataFrame to JSON format for the AI model
            clips_string = df_clips.to_json(orient="records", indent=4)    
            print(f"Generating script for summary video using {model}...\n")
            #add JSON formatting to the instructions
            instructions += """Return your answer as a JSON object with the format
            {"script":[
                        {'clip_path': path of the video clip file,
                        'narration': text of the narration for the clip in the summary video},...
                        ]
            }."""
            # Generate script using AI
            script = self.generate_text(
                prompt=instructions,
                instructions=clips_string,  
                model=model,
                output_type="json_object",
            )

            # Parse the generated script JSON safely
            script_json = json.loads(script)  # Use json.loads instead of ast.literal_eval

            # Ensure the key "script" exists in the response
            if "script" not in script_json:
                print("❌ Error: Response does not contain 'script' key.")
                return False

            # Convert JSON to DataFrame
            df_summary_script = pd.DataFrame(script_json["script"])

            return df_summary_script

        except json.JSONDecodeError:
            print("❌ Error: Failed to parse AI-generated script response as JSON.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        return False  # Return False if anything fails



    def generate_audio_narrations(self, df_summary_script, voice="nova", output_dir=None):
        """
        Generates audio narrations for each clip in the summary script DataFrame.
        The generated audio files are saved alongside the video clips unless a different output directory is specified.

        Parameters:
        ----------
        df_summary_script : pd.DataFrame
            DataFrame containing summary script details.
            Must contain columns: "clip_path" (video file path) and "narration" (text to convert to speech).
        voice : str, optional
            The voice to use for synthesis. Available voices:
            - 'nova' (default)
            - 'alloy'
            - 'echo'
            - 'fable'
            - 'onyx'
            - 'shimmer'
        output_dir : str, optional
            Directory where the audio files should be saved.
            If `None`, audio is saved next to the original video clips.

        Returns:
        -------
        bool
            Returns `True` if at least one audio narration is successfully generated, otherwise `False`.
        """

        # Ensure the necessary columns exist
        required_columns = {"clip_path", "narration"}
        if not required_columns.issubset(df_summary_script.columns):
            print(f"❌ Error: DataFrame must contain columns: {required_columns}")
            return False

        success_count = 0  # Track successful narrations

        for index, row in df_summary_script.iterrows():
            try:
                clip_path = row["clip_path"]
                narration = row["narration"]

                # Determine output path
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
                    audio_filename = os.path.basename(clip_path).replace(".mp4", ".mp3")
                    audio_path = os.path.join(output_dir, audio_filename)
                else:
                    audio_path = clip_path.replace(".mp4", ".mp3")

                # Generate audio narration
                if self.generate_audio(narration, audio_path, voice=voice):
                    print(f"✅ Audio narration created: {audio_path}")
                    success_count += 1
                else:
                    print(f"❌ Failed to generate audio for {clip_path}")

            except Exception as e:
                print(f"❌ Error processing {clip_path}: {e}")

        return success_count > 0  # Return True if at least one narration was generated


    def generate_summary_video(self, df_summary_script, file_path: str):
        """
        Combines audio narrations with processed video clips to create a final summary video.
        After successful creation, it deletes processed video clips and audio files.

        Parameters:
        ----------
        df_summary_script : pd.DataFrame
            DataFrame containing summary script details.
            Must contain columns: "clip_path" (video file path).
        file_path : str
            The path for the final output summary video.

        Returns:
        -------
        bool
            Returns `True` if the final video is successfully created, otherwise `False`.
        """

        # Ensure final video directory exists
        final_video_dir = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(final_video_dir, exist_ok=True)

        # Temporary file to store video list for concatenation
        with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".txt") as concat_list_file:
            concat_list_path = concat_list_file.name

        # List to store processed video clips
        processed_clips = []
        processed_audios = []  # Track processed audio files for cleanup

        # Process each video
        for index, row in df_summary_script.iterrows():
            video_path = os.path.abspath(row["clip_path"])  # Convert to absolute path
            audio_path = video_path.replace(".mp4", ".mp3")  # Corresponding audio file

            # Check if both video and audio files exist
            if not os.path.exists(video_path):
                print(f"❌ Missing video file: {video_path}, skipping...")
                continue
            if not os.path.exists(audio_path):
                print(f"❌ Missing audio file: {audio_path}, skipping...")
                continue

            # Define processed clip output path
            processed_clip = os.path.join(final_video_dir, f"processed_clip_{index:03d}.mp4")
            print(f"Proccessed clips will be saved in {processed_clip}")
            # FFmpeg command to replace audio and handle duration mismatches
            command = [
                self.ffmpeg_path,
                "-y",  # ✅ Forces overwrite to prevent FFmpeg from waiting for input
                "-i", video_path,        # Input video
                "-i", audio_path,        # Input audio
                "-map", "0:v:0",         # Use first video stream
                "-map", "1:a:0",         # Use first audio stream
                "-c:v", "libx264",       # Video codec
                "-preset", "ultrafast",  # Fast processing
                "-c:a", "aac",           # Audio codec
                "-b:a", "192k",          # High-quality audio bitrate
                "-strict", "experimental",
                "-shortest",             # Trim video if longer
                "-vf", "tpad=stop_mode=clone:stop_duration=5",  # Freeze last frame if audio is longer
                processed_clip
            ]

            try:
                #print(f"🎥 Processing clips and audio")
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"✅ Processed: {processed_clip}")
                processed_clips.append(processed_clip)
                processed_audios.append(audio_path)  # Track audio for deletion
            except subprocess.CalledProcessError as e:
                print(f"❌ Error processing {video_path}: {e.stderr.decode('utf-8')}")

        # Ensure there are clips to concatenate
        if not processed_clips:
            print("❌ No valid clips processed. Cannot create summary video.")
            return False

        # Write processed clips to concatenation list
        with open(concat_list_path, "w") as f:
            for clip in processed_clips:
                f.write(f"file '{clip.replace(os.sep, '/')}'\n")  # ✅ Works everywhere


        # FFmpeg command to concatenate processed clips into final video
        concat_command = [
            self.ffmpeg_path,
            "-y",  # ✅ Forces overwrite to prevent FFmpeg from waiting for input
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            file_path
        ]

        try:
            subprocess.run(concat_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"🎬 Final movie created: {file_path}")

            # ✅ Cleanup: Delete processed clips and audio files after successful video creation
            for clip in processed_clips:
                try:
                    os.remove(clip)
                    print(f"🗑️ Deleted processed clip: {clip}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {clip}: {e}")

            for audio in processed_audios:
                try:
                    os.remove(audio)
                    print(f"🗑️ Deleted processed audio: {audio}")
                except Exception as e:
                    print(f"⚠️ Failed to delete {audio}: {e}")

            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Error concatenating videos: {e.stderr.decode('utf-8')}")
            return False

        finally:
            # Cleanup temporary concat list file
            os.remove(concat_list_path)

