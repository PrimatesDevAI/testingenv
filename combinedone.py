import io
import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
import matplotlib
import matplotlib.pylab as plt
torchaudio.set_audio_backend("soundfile")
import pyaudio
global new_confidence1
global new_confidence2
new_confidence1=0
new_confidence2=0
import config


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
  
model = init_jit_model(model_path='./model.jit')

def validate(model,inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/abs_max
    sound = sound.squeeze() 
    return sound

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)
audio = pyaudio.PyAudio()

frames_to_record = 20 
frame_duration_ms = 250



continue_recording = True

import config
stream1 = audio.open(format=FORMAT,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK,
                input_device_index=1)
data1 = []
voiced_confidences1 = []
continue_recording1 = True

while continue_recording1:
    audio_chunk1 = stream1.read(int(SAMPLE_RATE * frame_duration_ms / 1000.0))
    data1.append(audio_chunk1)
    audio_int161 = np.frombuffer(audio_chunk1, np.int16)
    audio_float321 = int2float(audio_int161)
    vad_outs1 = validate(model, torch.from_numpy(audio_float321))
    new_confidence1 = vad_outs1[:,1].numpy()[0].item()
    config.nc1=new_confidence1
    voiced_confidences1.append(new_confidence1)
    #pp1.update(new_confidence1)
#pp1.finalize()


