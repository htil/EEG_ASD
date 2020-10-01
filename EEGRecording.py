from dataclasses import dataclass

@dataclass
class EEGRecording:
    '''Class for holding EEG data'''
    name: str
    freq: int
    chan: int
    clas: str
    data: object