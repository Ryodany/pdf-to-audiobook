import os
import os.path
import glob
import sys

import getopt

import PyPDF2

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf

from pydub import AudioSegment

import nltk
nltk.download("punkt_tab")

DEFAULT_SPEAKER = "speakers/sara_martin_eleven_labs.wav"

def create_wav_folder():
    current_dir = os.getcwd()

    wav_folder_path = os.path.join(current_dir, 'wav', 'wavs')
    if not os.path.exists(wav_folder_path):
        os.makedirs(wav_folder_path)

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

def get_pdf_files(dir: str):
    pdf_files = sorted(glob.glob(os.path.join(dir, "*.pdf")))

    if len(pdf_files) == 0:
        print(f"No pdf files found in {dir}")
        return None

    print("List of pdf files:")
    for index, file_path in enumerate(pdf_files):
        file_name = os.path.basename(file_path)
        print(f"{index + 1}. {file_name}")

    return pdf_files

def sentencize_with_spacy(text: str):
    import spacy

    #english = "en_core_web_sm"
    spanish = "es_core_news_lg"
    nlp = spacy.load(spanish)

    sentences = []
    for sentence in nlp(text).sents:
        sentences += sentence

    return sentences

def sentencize_with_nltk(text: str):
    tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    return tokenizer.sentences_from_text(text)

def text_to_speech(sentencized_text: list[str], tts_model: str, language: str | None, output_wav: str):
    model_version = tts_model.split("--")[-1]

    print()
    print(f"Transforming text to speech using model {model_version} ({tts_model}) into {output_wav}...")

    home_dir = os.path.expanduser("~")
    model_path = f"{home_dir}/.local/share/tts/{tts_model}"
    config = XttsConfig()
    config.load_json(f"{model_path}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    model.cuda()

    wav_files = []
    for index, text in enumerate(sentencized_text):
        wav_basename = os.path.splitext(os.path.basename(output_wav))[0]
        wav_file = f"wav/wavs/{wav_basename}_{index}.wav"
        wav_files.append(wav_file)

        outputs = model.synthesize(
            text,
            config,
            speaker_wav=DEFAULT_SPEAKER,
            #gpt_cond_len=3,
            language=language,
            )
        audio_data = outputs["wav"]
        sample_rate = 24000
        sf.write(wav_file, audio_data, sample_rate)

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)

    for wav_file in wav_files:
        print(f"wav_file: {wav_file}")
        audio = AudioSegment.from_wav(wav_file)
        combined += audio + silence

    combined.export(output_wav, format="wav")

    #print("Press to remove wav files (chunks)")
    #input()

    for wav_file in wav_files:
        os.remove(wav_file)

def voice_conversion(source_wav: str, speaker_wav: str, output_wav: str):
    tts_model = "voice_conversion_models/multilingual/vctk/freevc24"
    model_version = tts_model.split("/")[-1]

    print()
    print(f"Converting voice using model {model_version} ({tts_model}) with speaker {speaker_wav} into {output_wav}...")

    tts = TTS(model_name=tts_model, progress_bar=False).to("cuda")
    tts.voice_conversion_to_file(source_wav=source_wav, target_wav=speaker_wav, file_path=output_wav)

def pdf_to_mp3(pdf_files: list[str]):
    for pdf_file in pdf_files:
        with open(pdf_file, 'rb') as file:
            print(f"Extracting text from {pdf_file}...")
            reader = PyPDF2.PdfReader(file)
            sentencized_text = []
            for page in range(len(reader.pages)):
                pdf_text = reader.pages[page].extract_text()
                print(pdf_text)
                text = ""
                for index, char in enumerate(pdf_text):
                    if char == '\n' and index < len(pdf_text) - 1 and pdf_text[index + 1].isupper():
                        text += ". "
                    else:
                        text += char
                text = text.replace("\n", "")

                sentencized_text += [x for x in sentencize_with_nltk(text) if x != "."]
                print(sentencized_text)

        base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        wav_filename = f"wav/{base_filename}.wav"
        #intermediate_wav_filename = f"wav/_{base_filename}.wav"
        intermediate_wav_filename = wav_filename
        mp3_filename = f"mp3/{base_filename}.mp3"

        # Spanish model
        #text_to_speech(sentencized_text, "tts_models/es/css10/vits", None, intermediate_wav_filename)
        # multilingual model
        text_to_speech(sentencized_text, "tts_models--multilingual--multi-dataset--xtts_v2", "es", intermediate_wav_filename)
        print()
        print(f"Text to speech completed successfully, output within file {intermediate_wav_filename}")

        # there is no need for voice conversion, since the speaker is correctly applied in text_to_speech
        #voice_conversion(intermediate_wav_filename, DEFAULT_SPEAKER, wav_filename)

        print()
        print(f"Converting WAV ({wav_filename}) to MP3 ({mp3_filename})...")
        audio = AudioSegment.from_wav(wav_filename)
        audio = audio.set_frame_rate(44100)
        audio.export(mp3_filename, format="mp3")

        print(f'{mp3_filename} successfully created')
        print()

def help():
    script_name = os.path.basename(sys.argv[0])
    print(f"Usage: {script_name} (-f <file> | -d <dir>)")
    print("Options:")
    print("  -h, --help        Show this help message and exit")
    print("  -f, --file <file> PDF file (mutually exclusive with -d)")
    print("  -d, --dir <dir>   Directory with PDFs (mutually exclusive with -f)")

if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:d:", ["help", "file=", "dir="])

        pdf_files = []
        pdf_folder = ""

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                help()
                sys.exit()
            elif opt in ("-f", "--file"):
                pdf_files = [arg]
            elif opt in ("-d", "--dir"):
                pdf_folder = arg
    except getopt.GetoptError as err:
        print(f"Error: {err}")
        help()
        sys.exit(2)

    if pdf_folder != "":
        pdf_files = get_pdf_files(pdf_folder)
        if pdf_files is None:
            sys.exit(3)

    create_pdf_folder()
    create_wav_folder()
    create_mp3_folder()

    pdf_to_mp3(pdf_files)