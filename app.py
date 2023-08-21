from flask import Flask, render_template, request
import tensorflow as tf
import keras
from keras.models import load_model
import audioread
import soundfile as sf
from pydub import AudioSegment
import wave
import librosa
from pathlib import Path
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('DeepFakeDetection_CNN_MFCC_New.keras')

print('hi')
@app.route('/', methods = ["GET", "POST"])

def index():
    if request.method == "POST":
        file_audio = request.files['file']
        filename = secure_filename(file_audio.filename)
        file_audio.save(os.path.join('templates','audios',filename)) 
        f = checksform('templates/audios/'+filename)
        mfcc_list = []
        data, samplerate = librosa.load(f)
        data = data[:14800]
        S = librosa.feature.mfcc(y=data,sr=96000).real.T.flatten()
        mfcc_list.append(S)
        input_features = np.array(mfcc_list)
        input_features.shape[0]
        input_features = input_features.astype('float')
        input_features = np.reshape(input_features, (1, 20, -1))
        input_features = input_features[:, :, :, np.newaxis]
        predictions = model.predict(input_features, batch_size=1)
        # print(predictions)
        if (predictions >= 0.5):
          print("Fake")
        else:
          print("Real")
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)



def convert_single_to_wav(fil):
   wav_file = fil.split(".", 1)[0] + "." + "wav"
   audio = AudioSegment.from_mp3(fil)
   audio.export(wav_file, format="wav")
   return wav_file

# def extract_single_mfcc(filename, sr=96000):
#     mfcc_list = []
#     y = readAudioFile(filename)
#     S =  librosa.feature.mfcc(y=y,sr=sr).real.T.flatten()
#     mfcc_list.append(S)
#     return np.array(mfcc_list)

def checksform(fil):
    p = Path(fil)
    if (p.suffix) == '.map3':
       return (convert_single_to_wav(fil))
    else:
        return fil