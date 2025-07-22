import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from collections import Counter
import random
import pretty_midi
import io
import base64
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Additional imports for audio processing
try:
    import fluidsynth
    import scipy.io.wavfile
    from pydub import AudioSegment
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    st.warning("Audio processing libraries not available. Install fluidsynth and pydub for MP3 conversion.")

# Set page configuration
st.set_page_config(page_title="AI Music Generator", page_icon="🎵", layout="wide")

# Title and description
st.title("🎵 Simple AI Music Generator")
st.markdown("Upload MIDI files to train a simple neural network and generate new music!")

# Initialize session state
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'vocab' not in st.session_state:
    st.session_state.vocab = None
if 'vocab_size' not in st.session_state:
    st.session_state.vocab_size = 0
if 'last_generated_midi' not in st.session_state:
    st.session_state.last_generated_midi = None
if 'last_generated_notes' not in st.session_state:
    st.session_state.last_generated_notes = None

def parse_midi_file(file_content):
    """Parse MIDI file and extract note sequences"""
    try:
        midi_data = pretty_midi.PrettyMIDI(io.BytesIO(file_content))
        notes = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    # Simple representation: just the pitch
                    notes.append(note.pitch)
        
        return notes
    except Exception as e:
        st.error(f"Error parsing MIDI file: {e}")
        return []

def create_sequences(notes, sequence_length=50):
    """Create input sequences for training"""
    sequences = []
    targets = []
    
    for i in range(len(notes) - sequence_length):
        sequences.append(notes[i:i + sequence_length])
        targets.append(notes[i + sequence_length])
    
    return np.array(sequences), np.array(targets)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, LayerNormalization

def build_model(vocab_size, sequence_length):
    """Build an improved LSTM model with bidirectional layers and layer normalization"""

    inputs = Input(shape=(sequence_length,))
    x = Embedding(vocab_size, 128)(inputs)
    
    # First Bidirectional LSTM with return_sequences=True
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM
    x = Bidirectional(LSTM(256))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def generate_music(model, vocab, sequence_length, num_notes=100, temperature=1.0):
    """Generate new music using the trained model"""
    # Start with a random sequence from the vocabulary
    start_sequence = [random.choice(list(vocab.keys())) for _ in range(sequence_length)]
    generated = start_sequence.copy()
    
    for _ in range(num_notes):
        # Prepare input sequence
        input_seq = np.array([vocab[note] for note in start_sequence[-sequence_length:]])
        input_seq = input_seq.reshape(1, sequence_length)
        
        # Predict next note
        prediction = model.predict(input_seq, verbose=0)[0]
        
        # Apply temperature for creativity
        prediction = np.log(prediction + 1e-7) / temperature
        exp_prediction = np.exp(prediction)
        prediction = exp_prediction / np.sum(exp_prediction)
        
        # Sample from the distribution
        next_note_idx = np.random.choice(len(prediction), p=prediction)
        next_note = [note for note, idx in vocab.items() if idx == next_note_idx][0]
        
        generated.append(next_note)
        start_sequence.append(next_note)
    
    return generated

def create_midi_from_notes(notes, output_file='generated_music.mid'):
    """Create MIDI file from note sequence"""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0
    note_duration = 0.5  # Half second per note
    
    for note_pitch in notes:
        if 0 <= note_pitch <= 127:  # Valid MIDI pitch range
            note = pretty_midi.Note(
                velocity=64,
                pitch=int(note_pitch),
                start=current_time,
                end=current_time + note_duration
            )
            instrument.notes.append(note)
            current_time += note_duration
    
    midi.instruments.append(instrument)
    return midi

def midi_to_audio(midi_obj, sample_rate=44100):
    """Convert MIDI to audio using fluidsynth"""
    if not AUDIO_LIBS_AVAILABLE:
        return None
    
    try:
        # Synthesize audio from MIDI
        audio = midi_obj.synthesize(fs=sample_rate)
        return audio
    except Exception as e:
        st.error(f"Error converting MIDI to audio: {e}")
        return None

def create_mp3_from_midi(midi_obj, filename='generated_music.mp3'):
    """Convert MIDI to MP3"""
    if not AUDIO_LIBS_AVAILABLE:
        return None
    
    try:
        # Convert MIDI to audio
        audio = midi_to_audio(midi_obj)
        if audio is None:
            return None
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32767).astype(np.int16)
        
        # Save as WAV first
        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, 44100, audio)
        wav_buffer.seek(0)
        
        # Convert WAV to MP3
        audio_segment = AudioSegment.from_wav(wav_buffer)
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)
        
        return mp3_buffer.getvalue()
    except Exception as e:
        st.error(f"Error creating MP3: {e}")
        return None

def get_midi_download_link(midi_obj, filename):
    """Create download link for MIDI file"""
    midi_bytes = io.BytesIO()
    midi_obj.write(midi_bytes)
    midi_bytes.seek(0)
    b64 = base64.b64encode(midi_bytes.read()).decode()
    href = f'<a href="data:audio/midi;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def get_mp3_download_link(mp3_bytes, filename):
    """Create download link for MP3 file"""
    b64 = base64.b64encode(mp3_bytes).decode()
    href = f'<a href="data:audio/mp3;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def create_simple_midi_player(midi_obj):
    """Create a simple MIDI player using HTML5 audio with converted audio"""
    if not AUDIO_LIBS_AVAILABLE:
        st.warning("Audio playback not available. Install fluidsynth and pydub for audio playback.")
        return
    
    audio = midi_to_audio(midi_obj)
    if audio is None:
        st.error("Could not convert MIDI to audio for playback")
        return
    
    # Normalize and convert to WAV
    audio = audio / np.max(np.abs(audio))
    audio = (audio * 32767).astype(np.int16)
    
    wav_buffer = io.BytesIO()
    scipy.io.wavfile.write(wav_buffer, 44100, audio)
    wav_buffer.seek(0)
    
    # Create base64 encoded audio
    audio_b64 = base64.b64encode(wav_buffer.getvalue()).decode()
    
    # HTML5 audio player
    audio_html = f"""
    <audio controls style="width: 100%;">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
    
    st.markdown(audio_html, unsafe_allow_html=True)

# Sidebar for controls
st.sidebar.header("Controls")

# File upload section
st.header("1. Upload MIDI Files")
uploaded_files = st.file_uploader(
    "Choose MIDI files", 
    accept_multiple_files=True, 
    type=['mid', 'midi']
)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} MIDI files")
    
    # Parse uploaded files
    all_notes = []
    file_info = []
    
    for file in uploaded_files:
        file_content = file.read()
        notes = parse_midi_file(file_content)
        if notes:
            all_notes.extend(notes)
            file_info.append({
                'filename': file.name,
                'notes_count': len(notes),
                'pitch_range': f"{min(notes)}-{max(notes)}" if notes else "N/A"
            })
    
    # Display file information
    if file_info:
        st.subheader("File Information")
        df = pd.DataFrame(file_info)
        st.dataframe(df)
        
        st.info(f"Total notes extracted: {len(all_notes)}")
        
        # Training section
        st.header("2. Train Model")
        
        col1, col2 = st.columns(2)
        with col1:
            sequence_length = st.slider("Sequence Length", min_value=10, max_value=100, value=50)
            epochs = st.slider("Training Epochs", min_value=1, max_value=50, value=10)
        
        with col2:
            batch_size = st.slider("Batch Size", min_value=16, max_value=128, value=32)
            
        if st.button("Train Model", type="primary"):
            if len(all_notes) < sequence_length + 1:
                st.error("Not enough notes to create training sequences. Upload more MIDI files.")
            else:
                with st.spinner("Training model..."):
                    # Create vocabulary
                    unique_notes = sorted(set(all_notes))
                    vocab = {note: i for i, note in enumerate(unique_notes)}
                    vocab_size = len(vocab)
                    
                    # Create sequences
                    sequences, targets = create_sequences(all_notes, sequence_length)
                    
                    # Convert to model input format
                    X = np.array([[vocab[note] for note in seq] for seq in sequences])
                    y = to_categorical([vocab[note] for note in targets], num_classes=vocab_size)
                    
                    # Build and train model
                    model = build_model(vocab_size, sequence_length)
                    
                    # Training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    class StreamlitCallback(tf.keras.callbacks.Callback):
                        def on_epoch_end(self, epoch, logs=None):
                            progress = (epoch + 1) / epochs
                            progress_bar.progress(progress)
                            status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs['loss']:.4f}")
                    
                    # Train model
                    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
                    
                    model.fit(
                        X, y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2,
                        callbacks=[early_stopping, StreamlitCallback()],
                        verbose=0
                    )
                    
                    # Save to session state
                    st.session_state.trained_model = model
                    st.session_state.vocab = vocab
                    st.session_state.vocab_size = vocab_size
                    st.session_state.sequence_length = sequence_length
                    
                    st.success("Model trained successfully!")
                    progress_bar.empty()
                    status_text.empty()

# Generation section
if st.session_state.trained_model is not None:
    st.header("3. Generate Music")
    
    col1, col2 = st.columns(2)
    with col1:
        num_notes = st.slider("Number of notes to generate", min_value=50, max_value=500, value=100)
        temperature = st.slider("Creativity (temperature)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    
    with col2:
        st.info("Higher temperature = more creative/random")
        st.info("Lower temperature = more conservative/predictable")
    
    if st.button("Generate Music", type="primary"):
        with st.spinner("Generating music..."):
            generated_notes = generate_music(
                st.session_state.trained_model,
                st.session_state.vocab,
                st.session_state.sequence_length,
                num_notes,
                temperature
            )
            
            # Create MIDI file
            midi_obj = create_midi_from_notes(generated_notes)
            
            # Save to session state
            st.session_state.last_generated_midi = midi_obj
            st.session_state.last_generated_notes = generated_notes
            
            st.success("Music generated successfully!")

# Music Player and Download Section
if st.session_state.last_generated_midi is not None:
    st.header("4. Play and Download")
    
    # Music Player
    st.subheader("🎵 Music Player")
    create_simple_midi_player(st.session_state.last_generated_midi)
    
    # Download options
    st.subheader("📥 Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # MIDI Download
        st.markdown("**MIDI File:**")
        midi_download_link = get_midi_download_link(st.session_state.last_generated_midi, "generated_music.mid")
        st.markdown(midi_download_link, unsafe_allow_html=True)
    
    with col2:
        # MP3 Download
        st.markdown("**MP3 File:**")
        if AUDIO_LIBS_AVAILABLE:
            if st.button("Generate MP3", key="mp3_button"):
                with st.spinner("Converting to MP3..."):
                    mp3_bytes = create_mp3_from_midi(st.session_state.last_generated_midi)
                    if mp3_bytes:
                        mp3_download_link = get_mp3_download_link(mp3_bytes, "generated_music.mp3")
                        st.markdown(mp3_download_link, unsafe_allow_html=True)
                    else:
                        st.error("Failed to create MP3 file")
        else:
            st.info("MP3 conversion not available. Install required libraries.")
    
    # Show statistics
    st.subheader("📊 Generated Music Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Notes", len(st.session_state.last_generated_notes))
    with col2:
        st.metric("Unique Notes", len(set(st.session_state.last_generated_notes)))
    with col3:
        st.metric("Pitch Range", f"{min(st.session_state.last_generated_notes)}-{max(st.session_state.last_generated_notes)}")
    
    # Plot note distribution
    st.subheader("📈 Note Distribution")
    note_counts = Counter(st.session_state.last_generated_notes)
    df_notes = pd.DataFrame(list(note_counts.items()), columns=['Note', 'Count'])
    st.bar_chart(df_notes.set_index('Note'))

# Instructions
st.sidebar.markdown("---")
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. **Upload MIDI Files**: Choose one or more MIDI files to train on
2. **Train Model**: Adjust parameters and click train
3. **Generate Music**: Use the trained model to create new music
4. **Play & Download**: Listen to your creation and download as MIDI or MP3

**Tips:**
- More training data = better results
- Higher epochs = more training (but slower)
- Temperature controls creativity vs. similarity to training data

**Requirements for MP3:**
```bash
pip install fluidsynth
pip install pydub
```
""")

# Installation instructions
if not AUDIO_LIBS_AVAILABLE:
    st.sidebar.markdown("---")
    st.sidebar.error("**Audio Features Disabled**")
    st.sidebar.markdown("""
    To enable audio playback and MP3 export:
    
    ```bash
    pip install fluidsynth
    pip install pydub
    ```
    
    You may also need to install FluidSynth system package:
    - **Ubuntu/Debian**: `sudo apt-get install fluidsynth`
    - **macOS**: `brew install fluidsynth`
    - **Windows**: Download from FluidSynth website
    """)