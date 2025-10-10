import streamlit as st
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import io
import os
from TTS.api import TTS
import tempfile
import gdown

st.set_page_config(page_title="Voice Cloner", page_icon="🎙️")

st.title("🎙️ Voice Cloning & Text-to-Speech")
st.write("Upload a voice sample and enter text to generate speech in that voice!")

# Check for pre-trained models
@st.cache_resource
def load_tts_model():
    try:
        # Using Coqui TTS for voice cloning
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
        return tts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # File upload
    audio_file = st.file_uploader("Upload voice sample (WAV/MP3, 5-30 seconds)", 
                                 type=['wav', 'mp3', 'ogg'])
    
    text_input = st.text_area("Enter text to generate speech:", 
                             "Hello! This is a test of voice cloning technology.")
    
    speaker_wav = None
    
    if audio_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            # Convert to WAV if needed
            if audio_file.name.endswith('.mp3'):
                audio, sr = librosa.load(audio_file, sr=22050)
                sf.write(tmp_file.name, audio, sr)
            else:
                tmp_file.write(audio_file.getvalue())
            speaker_wav = tmp_file.name
        
        # Play original audio
        st.audio(audio_file, caption="Original Voice Sample")
    
    if st.button("Generate Cloned Voice") and speaker_wav:
        with st.spinner("Generating speech... This may take a minute..."):
            try:
                tts = load_tts_model()
                if tts is None:
                    st.error("Failed to load TTS model")
                    return
                
                # Generate output
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
                    tts.tts_to_file(
                        text=text_input,
                        speaker_wav=speaker_wav,
                        language="en",
                        file_path=output_file.name
                    )
                    
                    # Play generated audio
                    st.audio(output_file.name, caption="Generated Speech")
                    
                    # Download button
                    with open(output_file.name, "rb") as file:
                        st.download_button(
                            label="Download Generated Audio",
                            data=file,
                            file_name="cloned_voice.wav",
                            mime="audio/wav"
                        )
                
                # Cleanup
                if os.path.exists(speaker_wav):
                    os.unlink(speaker_wav)
                if os.path.exists(output_file.name):
                    os.unlink(output_file.name)
                    
            except Exception as e:
                st.error(f"Error generating speech: {str(e)}")
                st.info("Try with a clearer voice sample (5-10 seconds of clean speech)")
    
    elif st.button("Generate Cloned Voice"):
        st.warning("Please upload a voice sample first!")

if __name__ == "__main__":
    main()
