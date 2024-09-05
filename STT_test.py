import argparse
import os
import numpy as np
import speech_recognition as sr
import whisper
import torch

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform

def STT():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    source = None

    if 'linux' in platform:
        mic_name = args.default_microphone
        print("mic name : ", mic_name)
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name == "pulse":
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
            if source is None:  # 마이크가 설정되지 않았을 경우
                raise ValueError(f"No matching microphone found for \"{mic_name}\".")
    else:
        source = sr.Microphone(sample_rate=16000)

    model = args.model
    if args.model != "tiny" and not args.non_english:
        model = model + ".en"

    # 모델을 GPU로 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        audio_model = whisper.load_model(model).to(device)
        if device == 'cuda':
            print("GPU loaded")
        else:
            print("CPU loaded")
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            print("CUDA out of memory error encountered. Switching to CPU.")
            device = 'cpu'
            audio_model = whisper.load_model(model).to(device)
        else:
            raise e

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    transcription = ['']

    if source is not None:
        with source:
            recorder.adjust_for_ambient_noise(source)
    else:
        raise RuntimeError("Microphone source is not initialized.")

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print("Model loaded.\n")

    while True:
        try:
            now = datetime.now()
            if not data_queue.empty():
                phrase_complete = False
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                phrase_time = now
                
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # 입력 데이터를 GPU로 이동
                audio_np = torch.tensor(audio_np).to(device)
                print("audio data loaded : ", datetime.now())

                # STT 수행 (GPU 강제 사용)
                result = audio_model.transcribe(audio_np, fp16=(device == 'cuda'))

                text = result['text'].strip()
                print("text generated : ", datetime.now())

                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text

                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)
        except KeyboardInterrupt:
            break

    return text

STT()