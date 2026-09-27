"""
Custom Audio Recorder with PyAudio
High-quality recording with real-time monitoring and noise reduction
"""

import numpy as np
import pyaudio
import wave
import threading
import time
from pathlib import Path
import tempfile

class AudioRecorder:
    """
    Custom audio recorder with real-time level monitoring
    """
    
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paFloat32
        
        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.max_level = 0
        self.current_level = 0
        self.recording_thread = None
        
        # List available microphones
        self.microphones = self._list_microphones()
        self.selected_mic = 0
    
    def _list_microphones(self):
        """List all available input devices"""
        mics = []
        info = self.audio.get_device_count()
        
        for i in range(info):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0:
                mics.append({
                    'index': i,
                    'name': device_info.get('name'),
                    'channels': device_info.get('maxInputChannels'),
                    'sample_rate': int(device_info.get('defaultSampleRate'))
                })
        
        return mics
    
    def select_microphone(self, mic_index):
        """Select microphone by index"""
        if 0 <= mic_index < len(self.microphones):
            self.selected_mic = mic_index
            return True
        return False
    
    def get_microphone_name(self):
        """Get name of selected microphone"""
        if self.microphones:
            return self.microphones[self.selected_mic]['name']
        return "Default"
    
    def _recording_callback(self):
        """Callback for recording thread"""
        try:
            mic_info = self.microphones[self.selected_mic] if self.microphones else None
            device_index = mic_info['index'] if mic_info else None
            
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                exceptions=False
            )
            
            self.frames = []
            
            while self.is_recording:
                try:
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    self.frames.append(data)
                    
                    # Calculate audio level
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.current_level = np.sqrt(np.mean(audio_data ** 2))
                    self.max_level = max(self.max_level, self.current_level)
                    
                except Exception as e:
                    print(f"Recording error: {e}")
                    break
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
        
        except Exception as e:
            print(f"Stream error: {e}")
    
    def start_recording(self):
        """Start recording audio"""
        if not self.is_recording:
            self.is_recording = True
            self.frames = []
            self.max_level = 0
            self.current_level = 0
            
            self.recording_thread = threading.Thread(target=self._recording_callback, daemon=True)
            self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording and return audio file path"""
        if self.is_recording:
            self.is_recording = False
            
            # Wait for thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=2)
            
            # Save to temporary file
            if self.frames:
                temp_dir = tempfile.gettempdir()
                output_path = str(Path(temp_dir) / "patient_audio_raw.wav")
                
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(self.channels)
                    wav_file.setsampwidth(self.audio.get_sample_size(self.format))
                    wav_file.setframerate(self.sample_rate)
                    wav_file.writeframes(b''.join(self.frames))
                
                return output_path
        
        return None
    
    def get_recording_duration(self):
        """Get duration of recording in seconds"""
        if self.frames:
            total_samples = sum(len(f) // 4 for f in self.frames)  # 4 bytes per float32 sample
            return total_samples / self.sample_rate
        return 0
    
    def get_audio_level(self):
        """Get current audio level (0.0 to 1.0)"""
        return min(self.current_level * 5, 1.0)  # Scale for display
    
    def get_max_level(self):
        """Get maximum audio level during recording"""
        return min(self.max_level * 5, 1.0)
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
        
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
