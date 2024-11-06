from pydub import AudioSegment
from pydub.playback import play

class ErrorHandler:
    def __init__(self):
        pass
    
    def onError():
        song = AudioSegment.from_wav("../../assets/aud/Utils/Error.wav")
        play(song)