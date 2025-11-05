import streamlit as st
import tempfile
import os
import io

st.set_page_config(page_title="Voice Cloner", page_icon="🎙️")

st.title("🎙️ Voice Cloning & Text-to-Speech")
st.write("⚠️ **Note:** Full voice cloning requires powerful GPU. For mobile deployment, we'll use a pre-trained voice.")

def load_tts_model():
    """Load TTS model with error handling"""
    try:
        from TTS.api import TTS
        # Use a pre-trained model that doesn't require voice cloning
        tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph")
        return tts
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def main():
    st.sidebar.header("Settings")
    voice_option = st.sidebar.selectbox(
        "Choose Voice Type",
        ["Pre-trained Female Voice", "Pre-trained Male Voice"]
    )
    
    text_input = st.text_area(
        "Enter text to generate speech:",
        "Hello! This is a demonstration of text to speech technology.",
        height=100
    )
    
    if st.button("Generate Speech"):
        with st.spinner("Loading model and generating speech..."):
            try:
                tts = load_tts_model()
                if tts is None:
                    st.error("Failed to load TTS model. This might be due to memory limitations.")
                    return
                
                # Generate speech
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as output_file:
                    tts.tts_to_file(text=text_input, file_path=output_file.name)
                    
                    # Display audio
                    audio_bytes = open(output_file.name, 'rb').read()
                    st.audio(audio_bytes, format='audio/wav')
                    
                    # Download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="generated_speech.wav",
                        mime="audio/wav"
                    )
                
                # Cleanup
                os.unlink(output_file.name)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("This might be due to limited resources on Streamlit Cloud. Try shorter text.")

if __name__ == "__main__":
    main()
