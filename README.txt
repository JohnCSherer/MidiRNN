Ensure that tensorflow, numpy and py_midicsv are installed.

Run the converter.py script to convert a MIDI file into a model-readable transcription.
arg1: path for the MIDI file, several are included
arg2: number of scale degrees to transpose the piece (model is trained to compose in C 
      major, so if the midi file is in D, for example, this value should be -2)

Running the model.py script will train the model on the last converted MIDI file and
print an accuracy measurement.