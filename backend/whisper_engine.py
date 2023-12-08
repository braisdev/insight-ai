from openai import OpenAI
import io
from pydub import AudioSegment
import time
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

client = OpenAI()

AUDIO_FILE_EXT_ALIASES = {
    "m4a": "mp4",
    "wave": "wav",
    # Add other aliases as needed
}

def get_text_chunks(raw_text):
    """
    Get the text chunks from the raw text
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(raw_text)

    return text_chunks

def transcribe_audio(uploaded_files):

    combined_transcripts = []  # List to hold transcripts from all files

    for uploaded_file in uploaded_files:
        # Determine the file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()

        # Map the file extension to the format, using the aliases dictionary
        audio_format = AUDIO_FILE_EXT_ALIASES.get(file_extension, file_extension)

        # Check if the format is supported
        supported_formats = {'mp3', 'wav', 'ogg', 'flv', 'mp4'}  # Add more supported formats as needed
        if audio_format not in supported_formats:
            raise ValueError(f"Unsupported file format: {audio_format}")

        # Load the audio file from the uploaded file object
        audio = AudioSegment.from_file(uploaded_file, format=audio_format)

        # Define the duration of each chunk in minutes
        chunk_duration = 20  # Modify as needed
        chunk_duration_ms = chunk_duration * 60 * 1000

        transcripts = []  # List to hold transcripts for the current file

        # Split the audio into chunks
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            chunk = audio[i: i + chunk_duration_ms]
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            file_obj.name = uploaded_file.name.split('.')[0] + f"_part_{split_number}.mp3"

            # Transcribe
            print(f"Transcribing part {split_number + 1} of {file_obj.name}")
            attempts = 0
            while attempts < 3:
                try:
                    transcript = client.audio.transcriptions.create(model="whisper-1", file=file_obj, response_format="text")
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed. Exception: {str(e)}")
                    time.sleep(5)
            else:
                print("Failed to transcribe after 3 attempts.")
                continue

            # Assuming Document is a defined function or class
            doc = Document(
                page_content=transcript,
                metadata={"source": file_obj.name, "chunk": split_number},
            )

            transcripts.append(doc)

        # Combine transcripts for the current file
        combined_text = " ".join([doc.page_content for doc in transcripts])
        combined_transcripts.append(combined_text)

    # Combine all transcripts from all files
    final_transcript = " ".join(combined_transcripts)

    # Assuming get_text_chunks is a defined function
    text_chunks = get_text_chunks(final_transcript)

    return text_chunks