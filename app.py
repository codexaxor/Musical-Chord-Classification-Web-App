import pickle
import soundfile as sf
import librosa
import numpy as np
import streamlit as st
# import matplotlib.pyplot as plt

st.title('Musical Chord Classification: Major and Minor')

def parse_audio(x):
    return x.flatten('F')[:x.shape[0]] 

def mean_mfccs(x):
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

def play_music(input_path):
    audio_file = open(input_path, 'rb')
    audio_bytes = audio_file.read()
    return audio_bytes

# def display_music(x):
#     fig = plt.figure(figsize=(1,1))
#     plt.plot(x)
#     return fig

loaded_model = pickle.load(open('music_model.sav', 'rb'))

st.write('Major chords are considered to sound happy but minor chords are considered to sound sad.')
input_path = ''

with st.form(key='my_form'):
    # input_path = st.text_input(label='Enter your music path')
    input_path = st.selectbox('Select Music',['musics/10_1.wav','musics/10_2.wav','musics/10_3.wav','musics/10_4.wav','musics/10_5.wav','musics/10_6.wav','musics/10_7.wav'])
    submit_button = st.form_submit_button(label='Submit')

if input_path and submit_button==True:
    try:
        with open(input_path):
            x, sr = sf.read(input_path, always_2d=True)
            x = parse_audio(x)
            z = mean_mfccs(x)
            z = np.array(z)
            result = loaded_model.predict(z.reshape(1,-1)).item()
            if result == 1:
                st.write('It\'s a *Major* Chord! :sunglasses: :guitar:')
                st.audio(input_path, format='audio/wav')
            else:
                st.write('It\'s a *Minor* Chord! :pensive: :guitar:')
                st.audio(input_path, format='audio/wav')

    except FileNotFoundError:
        st.error('No file in such directory.')

