"""
Enhanced Whisper Audio Handler with Noise Reduction
Supports audio quality improvement and background noise filtering
"""

import numpy as np
import whisper
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Global whisper model (loaded once)
_whisper_model = None

# Try to import audio processing libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_whisper_model():
    """
    Load Whisper model (lazy loading - only loads once)
    """
    global _whisper_model
    if _whisper_model is None:
        print("Loading Whisper model...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

def denoise_audio_librosa(audio_path, reduction_level=1):
    """
    Denoise audio using librosa spectral gating
    reduction_level: 0=off, 1=light, 2=medium, 3=heavy
    """
    if not LIBROSA_AVAILABLE or reduction_level == 0:
        return None
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        if reduction_level >= 1:
            # Light: Spectral subtraction
            S = librosa.stft(y)
            magnitude, phase = np.abs(S), np.angle(S)
            
            # Estimate noise from silent parts
            power = np.abs(magnitude) ** 2
            noise_power = np.percentile(power, 10, axis=1, keepdims=True)
            
            # Subtract noise
            clean_power = np.maximum(power - noise_power * (1 + reduction_level * 0.3), 0)
            clean_magnitude = np.sqrt(clean_power)
            
            # Reconstruct
            S_clean = clean_magnitude * np.exp(1j * phase)
            y_clean = librosa.istft(S_clean)
        else:
            y_clean = y
        
        # Save denoised audio
        output_path = str(audio_path).replace('.wav', '_denoised.wav')
        librosa.output.write_wav(output_path, y_clean, sr=16000)
        return output_path
    
    except Exception as e:
        print(f"Librosa denoising failed: {e}")
        return None

def denoise_audio_scipy(audio_path, reduction_level=1):
    """
    Denoise audio using scipy signal processing
    reduction_level: 0=off, 1=light, 2=medium, 3=heavy
    """
    if not SCIPY_AVAILABLE or reduction_level == 0:
        return None
    
    try:
        import soundfile as sf
        
        # Read audio
        audio, sr = sf.read(audio_path)
        
        if len(audio.shape) > 1:
            audio = audio[:, 0]  # Mono
        
        # Apply high-pass filter to remove rumble/hum
        sos = signal.butter(4, 80, 'hp', fs=sr, output='sos')
        audio_filtered = signal.sosfilt(sos, audio)
        
        # Apply noise gate (reduce amplitude of quiet parts)
        threshold = np.percentile(np.abs(audio_filtered), 10)
        audio_gated = np.where(np.abs(audio_filtered) > threshold * (1 + reduction_level * 0.2), audio_filtered, 0)
        
        # Normalize
        max_val = np.max(np.abs(audio_gated))
        if max_val > 0:
            audio_gated = audio_gated / max_val * 0.95
        
        # Save
        output_path = str(audio_path).replace('.wav', '_filtered.wav')
        sf.write(output_path, audio_gated, sr)
        return output_path
    
    except Exception as e:
        print(f"Scipy denoising failed: {e}")
        return None

def simple_noise_gate(audio_path, reduction_level=1):
    """
    Simple noise gate without external dependencies
    """
    try:
        import wave
        
        # Read audio
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        # Apply noise gate
        threshold = np.percentile(np.abs(audio), 15)
        gate_amount = 1.0 + (reduction_level * 0.15)
        audio_gated = np.where(
            np.abs(audio) > threshold * gate_amount,
            audio,
            0
        )
        
        # Smooth transitions
        if reduction_level > 0:
            kernel_size = 5 + reduction_level * 10
            kernel = np.hanning(kernel_size) / kernel_size
            audio_gated = np.convolve(audio_gated, kernel, mode='same')
        
        # Convert back to int16
        audio_output = (audio_gated * 32767).astype(np.int16)
        
        # Save
        output_path = str(audio_path).replace('.wav', '_denoised_simple.wav')
        with wave.open(output_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_output.tobytes())
        
        return output_path
    
    except Exception as e:
        print(f"Simple denoising failed: {e}")
        return None

# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe_audio_improved(audio_path, noise_reduction_level=1):
    """
    Transcribe audio with optional noise reduction
    noise_reduction_level: 0=off, 1=light, 2=medium, 3=heavy
    
    Args:
        audio_path: Path to audio file
        noise_reduction_level: 0-3, higher = more aggressive filtering
    
    Returns:
        Transcribed text
    """
    
    # Step 1: Try noise reduction
    processed_path = audio_path
    
    if noise_reduction_level > 0:
        print(f"Applying noise reduction (level {noise_reduction_level})...")
        
        # Try librosa first (best quality)
        if LIBROSA_AVAILABLE:
            processed_path = denoise_audio_librosa(audio_path, noise_reduction_level)
            if processed_path:
                print("✓ Noise reduction applied (librosa)")
        
        # Fall back to scipy
        if not processed_path or not Path(processed_path).exists():
            if SCIPY_AVAILABLE:
                processed_path = denoise_audio_scipy(audio_path, noise_reduction_level)
                if processed_path:
                    print("✓ Noise reduction applied (scipy)")
        
        # Fall back to simple gate
        if not processed_path or not Path(processed_path).exists():
            processed_path = simple_noise_gate(audio_path, noise_reduction_level)
            if processed_path:
                print("✓ Noise reduction applied (simple gate)")
        
        # If all fail, use original
        if not processed_path or not Path(processed_path).exists():
            print("⚠ Noise reduction failed, using original audio")
            processed_path = audio_path
    
    # Step 2: Load Whisper model
    model = load_whisper_model()
    
    # Step 3: Transcribe
    print("Transcribing audio...")
    result = model.transcribe(processed_path, language="en")
    
    transcription = result["text"].strip()
    
    # Clean up temporary files
    if processed_path != audio_path:
        try:
            Path(processed_path).unlink()
        except:
            pass
    
    return transcription if transcription else "Could not transcribe audio. Please speak more clearly."


def transcribe_audio(audio_path):
    """
    Simple transcription without noise reduction (legacy support)
    """
    return transcribe_audio_improved(audio_path, noise_reduction_level=0)


def record_audio_streamlit():
    """
    Placeholder for Streamlit audio recording
    (handled by st.audio_input in the app)
    """
    return None
