"""
External Open-Access Speech Enhancement Datasets Guide & Fetcher
=================================================================
Provides instructions and utility scripts to download standard public
academic speech enhancement datasets:

1. NOIZEUS: A Noisy Speech Corpus for Evaluation of Speech Enhancement Algorithms
   - Standard benchmark created by Dr. Philipos C. Loizou (UT Dallas)
   - IEEE sentences corrupted by 8 real-world noises at 0, 5, 10, 15 dB SNR
   - URL: https://ecs.utdallas.edu/loizou/speech/noizeus/

2. LibriSpeech Clean Speech Dataset:
   - Thousands of hours of 16 kHz read English speech
   - Open source, public domain (CC BY 4.0)
   - URL: https://www.openslr.org/12

3. DEMAND: Diverse Environments Multi-channel Acoustic Noise Database
   - High-quality 16-channel recordings of real environmental acoustic noise
   - Categories: Office, Street, Nature, Transportation, Domestic
   - URL: https://zenodo.org/records/1227121

4. ESC-50 (Environmental Sound Classification):
   - 2,000 environmental recordings of 50 classes of natural sounds
   - URL: https://github.com/karolpiczak/ESC-50
"""

import sys
import urllib.request
from pathlib import Path


def print_public_dataset_guide():
    guide = """
================================================================================
   STANDARD PUBLIC ECE DATASETS FOR SPEECH ENHANCEMENT & NOISE REDUCTION
================================================================================

1. NOIZEUS Corpus (Most widely cited in IEEE / ECE research):
   - Description: 30 IEEE sentences produced by 3 male and 3 female speakers.
   - Corrupted by 8 real noises: Airport, Babble, Car, Restaurant, Street, Train, 
     Train Station, and Exhibition Hall at SNR levels: 0 dB, 5 dB, 10 dB, 15 dB.
   - Access: https://ecs.utdallas.edu/loizou/speech/noizeus/

2. OpenSLR LibriSpeech:
   - Description: Clean speech corpus widely used for speech recognition & enhancement.
   - Access: https://www.openslr.org/12/
   - Mini sample: http://www.openslr.org/resources/12/dev-clean.tar.gz

3. DEMAND Noise Database:
   - Description: 16-channel recordings of 18 diverse real-world noise environments.
   - Access: https://zenodo.org/records/1227121

4. Microsoft DNS Challenge (Deep Noise Suppression):
   - Description: Over 500 hours of clean speech and 100+ hours of real noise.
   - Access: https://github.com/microsoft/DNS-Challenge

HOW TO USE EXTERNAL AUDIO FILES WITH THIS PROJECT:
1. Place any downloaded .wav or .mp3 audio file into the 'dataset/' or 'sample_data/' folder,
   or simply drag-and-drop it into the 'Upload Audio' tab in the Streamlit application!
================================================================================
"""
    print(guide)


if __name__ == "__main__":
    print_public_dataset_guide()
