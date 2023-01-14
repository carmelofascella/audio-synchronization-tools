# Audio Synchronization tools

This repository hosts the scripts used to synchronize different representation of the same song (audio or symbolic files).

The algorithm is based on the paper <a href="https://ieeexplore.ieee.org/abstract/document/4959972" target="_blank">High resolution audio synchronization using chroma onset features</a>.

This repo was written during my internship in International Audio Labs in Erlangen in 2020/2021.
The data folder is not available for privacy reasons.

## 1. Setup

`pip install -r requirements.txt `

## 2. Execution

`audio-audio synchronization.ipynb`: synchronization of two audio files containing the same song but played in different ways.

`audio-symbolic synchronization.ipynb`: synchronization an audio file and a symbolic file containing the same song but played in different ways.

## 3. Problem

In recent years, the availability of numerous digital music libraries and streaming services has made it possible to find different versions and representations of a certain piece of music.

A musical work can be digitally represented as an audio recording file (i.e. .WAV, .MP3), a symbolic score (i.e. MIDI, MusicXML, ) or a digitized sheet music transcription (score images).

These formats differ fundamentally in their respective structures and content, but they contain the same musical information.

Furthermore, a certain musical piece can be played and interpreted by different artists, leading to various versions of the song, called cover song. A cover song may differ from the original one with respect of the instruments, it may represent a different genre, it can have a different arrangement, it can be played in a different tuning, or it may present a different musical structure.

Another important difference to consider when analysing two different audio recordings referring to the same piece of music is the environment in which they are recorded and the different quality that the recordings may have.

The general task of music synchronization is to automatically link the various data streams referred to the same musical information. Audio and symbolic files are stored in such a way that they can be associated with a timeline, which can be expressed in physical time, such as seconds, or musical time, such as measures or beats. In score images we refer to spatial positions rather then time positions, and each musical event can be associated to a certain pixel coordinate.

After extracting two feature sequences from the two musical representation that we want to align, we need to introduce a distance anchor point parameter $\tau$, that sets up local consecutive constraint regions, corresponding to a certain number of musical measures.

In this way we can compute a local alignment between sub-sequences of features that contain the same musical information.

Each local alignment is computed by applying standard DTW algorithm.

The computation of the local alignments are independent of each other, and for this reason the are computed in a sequential way.
