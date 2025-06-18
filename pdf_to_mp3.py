import os
import os.path
import glob
import sys
import shutil
from datetime import datetime

import getopt
from loguru import logger

from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    ProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    Task,
    filesize,
)
from rich.text import Text

import PyPDF2

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf

from pydub import AudioSegment

import nltk

nltk.download("punkt_tab")

DEFAULT_SPEAKER = "speakers/sara_martin_eleven_labs.wav"


class RateColumn(ProgressColumn):
    """Renders human readable processing rate."""

    def render(self, task: "Task") -> Text:
        """Render the speed in iterations per second."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("", style="progress.percentage")
        unit, suffix = filesize.pick_unit_and_suffix(
            int(speed),
            ["", "x10³", "x10⁶", "x10⁹", "x10¹²"],
            1000,
        )
        data_speed = speed / unit

        if data_speed < 1:
            seconds_per_iter = 1 / speed
            return Text(f"{seconds_per_iter:.3f} s/it", style="progress.percentage")
        else:
            return Text(f"{data_speed:.1f}{suffix} it/s", style="progress.percentage")


def get_logging_folder():
    current_dir = os.getcwd()
    return os.path.join(
        current_dir, "logs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )


def create_wav_folder():
    current_dir = os.getcwd()

    wav_folder_path = os.path.join(current_dir, "wav", "wavs")
    shutil.rmtree(wav_folder_path)
    os.makedirs(wav_folder_path)


def create_mp3_folder():
    current_dir = os.getcwd()

    mp3_folder_path = os.path.join(current_dir, "mp3")
    if not os.path.exists(mp3_folder_path):
        os.makedirs(mp3_folder_path)


def create_pdf_folder():
    current_dir = os.getcwd()

    pdf_folder_path = os.path.join(current_dir, "pdf")
    if not os.path.exists(pdf_folder_path):
        os.makedirs(pdf_folder_path)


def create_logging_folder():
    logs_folder_path = get_logging_folder()
    if not os.path.exists(logs_folder_path):
        os.makedirs(logs_folder_path)


def create_logger(pdf_filename: str):
    log_dir = get_logging_folder()
    log_file = f"{log_dir}/{pdf_filename}.log"

    def pdf_filter(record):
        return record["extra"].get("pdf_name") == pdf_filename

    return logger.add(
        log_file,
        format="{time} | {level} | {message}",
        colorize=True,
        filter=pdf_filter,
    )


def get_logger(pdf_filename: str):
    return logger.bind(pdf_name=pdf_filename)


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


def sentencize_with_nltk(text: str):
    tokenizer = nltk.data.load("tokenizers/punkt/spanish.pickle")
    return tokenizer.sentences_from_text(text)


def text_to_speech(
    progress: Progress,
    sentencized_text: list[str],
    tts_model: str,
    language: str | None,
    output_wav: str,
):
    model_version = tts_model.split("--")[-1]

    print()
    print(
        f"Transforming text to speech using model {model_version} ({tts_model}) into {output_wav}..."
    )

    home_dir = os.path.expanduser("~")
    model_path = f"{home_dir}/.local/share/tts/{tts_model}"
    config = XttsConfig()
    config.load_json(f"{model_path}/config.json")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
    model.cuda()

    wav_files = []
    text_to_speech_task = progress.add_task(
        description="Converting sentences into audio", current=""
    )
    for index, text in progress.track(
        enumerate(sentencized_text),
        task_id=text_to_speech_task,
        total=len(sentencized_text),
    ):
        wav_basename = os.path.splitext(os.path.basename(output_wav))[0]
        wav_file = f"wav/wavs/{wav_basename}_{index}.wav"
        wav_files.append(wav_file)

        outputs = model.synthesize(
            text,
            config,
            speaker_wav=DEFAULT_SPEAKER,
            # gpt_cond_len=3,
            language=language,
        )
        audio_data = outputs["wav"]
        sample_rate = 24000
        sf.write(wav_file, audio_data, sample_rate)

    combined = AudioSegment.empty()
    silence = AudioSegment.silent(duration=500)

    wav_to_mp3_task = progress.add_task(
        description="Converting wav into mp3", current=""
    )
    for wav_file in progress.track(
        wav_files, task_id=wav_to_mp3_task, total=len(wav_files)
    ):
        wav_file_basename = os.path.splitext(os.path.basename(wav_file))[0]
        progress.update(wav_to_mp3_task, current=wav_file_basename)

        audio = AudioSegment.from_wav(wav_file)
        combined += audio + silence
        progress.update(wav_to_mp3_task, current=wav_file_basename)

    combined.export(output_wav, format="wav")

    for wav_file in wav_files:
        os.remove(wav_file)


def voice_conversion(source_wav: str, speaker_wav: str, output_wav: str):
    tts_model = "voice_conversion_models/multilingual/vctk/freevc24"
    model_version = tts_model.split("/")[-1]

    print()
    print(
        f"Converting voice using model {model_version} ({tts_model}) with speaker {speaker_wav} into {output_wav}..."
    )

    tts = TTS(model_name=tts_model, progress_bar=False).to("cuda")
    tts.voice_conversion_to_file(
        source_wav=source_wav, target_wav=speaker_wav, file_path=output_wav
    )


def pdf_to_mp3(pdf_files: list[str]):
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("[blue]{task.fields[current]}"),
        MofNCompleteColumn(),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        RateColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
    ) as progress:
        audiobooks_task = progress.add_task(
            description="Creating audiobooks", current=""
        )

        for pdf_file in progress.track(
            pdf_files, task_id=audiobooks_task, total=len(pdf_files)
        ):
            pdf_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            progress.update(audiobooks_task, current=pdf_filename)

            current_logger_sink_id = create_logger(pdf_filename)
            current_logger = get_logger(pdf_filename)

            with open(pdf_file, "rb") as file:
                current_logger.info(f"Extracting text from {pdf_file}...")

                reader = PyPDF2.PdfReader(file)
                sentencized_text = []
                extract_sentences_task = progress.add_task(
                    description="Extracting sentences from pages", current=""
                )
                for page in progress.track(
                    range(len(reader.pages)),
                    task_id=extract_sentences_task,
                    total=len(reader.pages),
                ):
                    pdf_text = reader.pages[page].extract_text()
                    current_logger.debug(f"Page {page}:")
                    current_logger.debug(pdf_text)
                    current_logger.debug("")

                    text = ""
                    for index, char in enumerate(pdf_text):
                        if (
                            char == "\n"
                            and index < len(pdf_text) - 1
                            and pdf_text[index + 1].isupper()
                        ):
                            text += ". "
                        else:
                            text += char
                    text = text.replace("\n", "")

                    sentencized_text += [
                        x for x in sentencize_with_nltk(text) if x != "."
                    ]
                    current_logger.debug(sentencized_text)
                    current_logger.debug("")

            base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            wav_filename = f"wav/{base_filename}.wav"
            # intermediate_wav_filename = f"wav/_{base_filename}.wav"
            intermediate_wav_filename = wav_filename
            mp3_filename = f"mp3/{base_filename}.mp3"

            # Spanish model
            # text_to_speech(progress, sentencized_text, "tts_models/es/css10/vits", None, intermediate_wav_filename)
            # multilingual model
            text_to_speech(
                progress,
                sentencized_text,
                "tts_models--multilingual--multi-dataset--xtts_v2",
                "es",
                intermediate_wav_filename,
            )
            current_logger.info("")
            current_logger.info(
                f"Text to speech completed successfully, output within file {intermediate_wav_filename}"
            )

            # there is no need for voice conversion, since the speaker is correctly applied in text_to_speech
            # voice_conversion(intermediate_wav_filename, DEFAULT_SPEAKER, wav_filename)

            current_logger.info("")
            current_logger.info(
                f"Converting WAV ({wav_filename}) to MP3 ({mp3_filename})..."
            )
            audio = AudioSegment.from_wav(wav_filename)
            audio = audio.set_frame_rate(44100)
            audio.export(mp3_filename, format="mp3")

            current_logger.info(f"{mp3_filename} successfully created")
            current_logger.info("")

            progress.update(audiobooks_task, current=pdf_filename)

            logger.remove(current_logger_sink_id)


def help():
    script_name = os.path.basename(sys.argv[0])
    print(f"Usage: {script_name} (-f <file> | -d <dir>)")
    print("Options:")
    print("  -h, --help        Show this help message and exit")
    print("  -f, --file <file> PDF file (mutually exclusive with -d)")
    print("  -d, --dir <dir>   Directory with PDFs (mutually exclusive with -f)")


def init_logging():
    logger.remove()

    create_logging_folder()


if __name__ == "__main__":
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

    init_logging()

    create_pdf_folder()
    create_wav_folder()
    create_mp3_folder()

    pdf_to_mp3(pdf_files)
