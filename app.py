import streamlit as st
import torch
import torchaudio
import librosa
import soundfile as sf
import numpy as np
import io
import os
import tempfile
import gdown
from pydub import AudioSegment

st.set_page_config(page_title="Voice Cloner", page_icon="🎙️")

st.title("🎙️ Voice Cloning & Text-to-Speech")
st.write("Upload a voice sample and enter text to generate speech in that voice!")

# Check for pre-trained models
@st.cache_resource
def load_tts_model():
    try:
        # Using Coqui TTS for voice cloning
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        
        # Initialize TTS with a voice cloning model
        tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(device)
        return tts
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def convert_audio(input_path, output_path):
    """Convert audio to WAV format with proper sampling rate"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(22050)
        audio = audio.set_channels(1)
        audio.export(output_path, format="wav")
        return True
    except Exception as e:
        st.error(f"Audio conversion error: {e}")
        return False

def main():
    st.sidebar.info("""
    **Instructions:**
    1. Upload a clear voice sample (5-30 seconds)
    2. Enter text to generate
    3. Click Generate button
    4. Wait for processing
    """)
    
    # File upload
    audio_file = st.file_uploader("Upload voice sample (WAV/MP3, 5-30 seconds)", 
                                 type=['wav', 'mp3', 'ogg', 'm4a'])
    
    text_input = st.text_area("Enter text to generate speech:", 
                             "Hello! This is a test of voice cloning technology.")
    
    speaker_wav = None
    
    if audio_file is not None:
        # Display audio info
        st.audio(audio_file, caption="Uploaded Voice Sample")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{audio_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            input_path = tmp_file.name
        
        # Convert to proper format
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as converted_file:
            if convert_audio(input_path, converted_file.name):
                speaker_wav = converted_file.name
                st.success("✅ Audio file processed successfully!")
            else:
                st.error("❌ Failed to process audio file")
        
        # Cleanup input file
        if os.path.exists(input_path):
            os.unlink(input_path)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🚀 Generate Cloned Voice", use_container_width=True):
            if not speaker_wav:
                st.warning("⚠️ Please upload a voice sample first!")
                return
            
            with st.spinner("🔄 Generating speech... This may take 1-2 minutes..."):
                try:
                    # Initialize TTS
                    from TTS.api import TTS
                    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts")
                    
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
                                label="📥 Download Generated Audio",
                                data=file,
                                file_name="cloned_voice.wav",
                                mime="audio/wav",
                                use_container_width=True
                            )
                    
                    st.success("✅ Voice generation completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating speech: {str(e)}")
                    st.info("💡 Try with a clearer voice sample (5-10 seconds of clean speech)")
    
    with col2:
        if st.button("🔄 Clear All", use_container_width=True):
            st.rerun()
    
    # Cleanup temporary files
    if speaker_wav and os.path.exists(speaker_wav):
        os.unlink(speaker_wav)

if __name__ == "__main__":
    main()
