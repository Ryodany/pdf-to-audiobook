import os
import os.path
import glob

import PyPDF2

import torch
from TTS.api import TTS

from pydub import AudioSegment

def create_mp3_folder():
    current_dir = os.getcwd()

    mp3_folder_path = os.path.join(current_dir, 'mp3')
    if not os.path.exists(mp3_folder_path):
        os.makedirs(mp3_folder_path)

def create_pdf_folder():
    current_dir = os.getcwd()

    pdf_folder_path = os.path.join(current_dir, 'pdf')
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)

def get_pdf_files():
    pdf_files = glob.glob('pdf/*.pdf')

    if len(pdf_files) == 0:
        print('No pdf files to convert')
        return None

    print('List of PDF files:')
    for index, file_path in enumerate(pdf_files):
        file_name = os.path.basename(file_path)
        print(f'{index + 1}. {file_name}')

    return pdf_files


# We get the file that needs to be converted and convert it according to the selected language
def get_pdf_file_and_convert(pdf_files):
    # Get the number of the file to be converted
    while True:
        file_number = input('Enter PDF file number to convert to MP3 or enter "0" to exit: ')
        try:
            file_number = int(file_number)
            if file_number == 0:
                return None
            if file_number < 1 or file_number > len(pdf_files):
                print(f'The number must be from 1 to {len(pdf_files)}')
            else:
                return pdf_files[file_number-1]
        except ValueError:
            print('Enter an integer')

def text_to_speech(text: str, tts_model: str, language: str | None, output_wav: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_version = tts_model.split("/")[-1]

    print()
    print(f"Transforming text to speech using model {model_version} ({tts_model}) into {output_wav}...")

    tts = TTS(tts_model, progress_bar=False).to(device)

    if language is not None:
        # for multilingual models, language must be specified
        tts.tts_to_file(text, speaker_wav="speakers/sara_martin_eleven_labs.wav", language="es", file_path=output_wav)
    else:
        # for monolingual models, language must not be specified
        tts.tts_to_file(text, file_path=output_wav)

def voice_conversion(source_wav: str, speaker_wav: str, output_wav: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tts_model = "voice_conversion_models/multilingual/vctk/freevc24"
    model_version = tts_model.split("/")[-1]

    print()
    print(f"Converting voice using model {model_version} ({tts_model}) with speaker {speaker_wav} into {output_wav}...")

    tts = TTS(model_name=tts_model, progress_bar=False).to(device)
    tts.voice_conversion_to_file(source_wav=source_wav, target_wav=speaker_wav, file_path=output_wav)

def pdf_to_mp3():
    create_pdf_folder()

    create_mp3_folder()

    pdf_files = get_pdf_files()
    if pdf_files is None:
        print("End of the program.")
        exit()

    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            print(f"Extracting text from {pdf_file}...")
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()

            base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            intermediate_wav_filename = f"wav/_{base_filename}.wav"
            wav_filename = f"wav/{base_filename}.wav"
            mp3_filename = f"mp3/{base_filename}.mp3"

            # Spanish model
            #text_to_speech(text, "tts_models/es/css10/vits", None, intermediate_wav_filename)
            # multilingual model
            #text_to_speech(text, "tts_models/multilingual/multi-dataset/xtts_v1.1", "es", intermediate_wav_filename)
            text_to_speech(text, "tts_models/multilingual/multi-dataset/xtts_v2", "es", intermediate_wav_filename)

            voice_conversion(intermediate_wav_filename, "speakers/sara_martin_eleven_labs.wav", wav_filename)

            print()
            print(f"Converting WAV ({wav_filename}) to MP3 ({mp3_filename})...")
            audio = AudioSegment.from_wav(wav_filename)
            audio = audio.set_frame_rate(44100)
            audio.export(mp3_filename, format="mp3")

            print(f'{mp3_filename} successfully created')
            print()


if __name__ == '__main__':
    pdf_to_mp3()