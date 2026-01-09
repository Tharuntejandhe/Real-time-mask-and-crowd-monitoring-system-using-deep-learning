# ==================== AUDIO_ALERT.PY ====================
"""
Audio Alert System
Provides text-to-speech alerts for crowd monitoring
"""

import pyttsx3
import threading
from config import ALERT_VOLUME, ALERT_RATE

class AudioAlert:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', ALERT_RATE)
        self.engine.setProperty('volume', ALERT_VOLUME)
        self.is_playing = False
    
    def play_alert(self, text, async_mode=True):
        """Play audio alert"""
        if self.is_playing:
            return False
        
        try:
            self.is_playing = True
            
            if async_mode:
                thread = threading.Thread(target=self._speak, args=(text,))
                thread.daemon = True
                thread.start()
            else:
                self._speak(text)
            
            return True
        except Exception as e:
            print(f"Error playing alert: {e}")
            self.is_playing = False
            return False
    
    def _speak(self, text):
        """Internal method to speak text"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        finally:
            self.is_playing = False
    
    def stop_alert(self):
        """Stop current alert"""
        try:
            self.engine.stop()
            self.is_playing = False
        except Exception as e:
            print(f"Error stopping alert: {e}")
    
    def set_voice(self, voice_id=0):
        """Set voice (0=default, varies by system)"""
        try:
            voices = self.engine.getProperty('voices')
            if voice_id < len(voices):
                self.engine.setProperty('voice', voices[voice_id].id)
        except Exception as e:
            print(f"Error setting voice: {e}")

# Global alert instance
_audio_alert = None

def get_alert_instance():
    """Get or create alert instance"""
    global _audio_alert
    if _audio_alert is None:
        _audio_alert = AudioAlert()
    return _audio_alert

def play_crowd_alert():
    """Play crowd alert message"""
    alert = get_alert_instance()
    message = "Warning! Excessive crowd detected. Please disperse and maintain social distancing."
    return alert.play_alert(message)

def play_custom_alert(message):
    """Play custom alert message"""
    alert = get_alert_instance()
    return alert.play_alert(message)
