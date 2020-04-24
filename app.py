# app.py
import os
import sys
import pygame
import pygame.midi
import time
import io
import copy
import random
#import midi
import mido
import pypianoroll
from pypianoroll import Multitrack, Track
from flask import Flask, flash, request, redirect, url_for
from pygame.locals import *
import numpy as np
from midiutil.MidiFile import MIDIFile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

UPLOAD_FOLDER = '/Users/donaldlleshi/Desktop/Capstone'
ALLOWED_EXTENSIONS = {'midi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model initialization
class SingleNoteRNN(nn.Module):
    def __init__(self, hidden_size, vocab_size=26, n_layers=1):
        super(SingleNoteRNN, self).__init__()
        self.vocab_size = vocab_size
        self.rnn = nn.GRU(vocab_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.ReLU = nn.LeakyReLU()

    def forward(self, data, hidden=None):
        # get the next output and hidden state
        output, hidden = self.rnn(data, hidden)
        # predict distribution over next tokens
        output = self.decoder(output)
        output = self.ReLU(output)
        return output, hidden



# END torch model setup

@app.route('/interface', methods=['GET'])
def interface_keyboard():

    return '''
    <!doctype html>
    <title>Interface Keyboard</title>
    <h1>Interface Keyboard</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form> '''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form> '''


# prints the id of the midi device connected
def print_devices():
    for n in range(pygame.midi.get_count()):
        print(n, pygame.midi.get_device_info(n))

def countdown(counter):
    now = time.time()
    timeout = now + counter
    countdown = counter
    while time.time() < timeout:
        if int(timeout - time.time()) == countdown:
            print(countdown)
            countdown = countdown - 1
    print("Play")




# reads input from a piano through polling and prints note number and velocity
def readInput(input_device):
    count = 0
    #print(time.time())
    timeout = time.time() + 5   # 5 seconds from now
    print("Reading input for 5 seconds")

    noteArr = []
    velocityArr = []
    timestampArr = []
    pairArr = []
    pendingArr = []

    pygame.fastevent.init()
    event_get = pygame.fastevent.get
    event_post = pygame.fastevent.post

    # poll for 5 seconds and generate and input midi file
    going = True

   
    eventArr = []
    print("instruction", "note", "MIDI number", "velocity", "timestamp")
    while True:
        if(time.time() > timeout):
            #print("breakpoll:", time.time())
            break
        events = event_get()
        for e in events:
            if e.type in [QUIT]:
                going = False
            if e.type in [KEYDOWN]:
                going = False
        
        if input_device.poll():
            midi_events = input_device.read(1)
            event = midi_events[0]




            data = event[0]
            timestamp = event[1]
            instruct = data[0] #144 for note on 128 for note off
            note_number = data[1]
            velocity = data[2]

            #for pairing note on/offs
            if note_number <= 73 and note_number >= 48:
                if instruct == 144:
                    pendingArr.append(event)
                elif instruct == 128:
                    for pend in pendingArr:
                        if note_number == pend[0][1]:
                            pairArr.append((pend, event))
                            pendingArr.remove(pend)
                            break
                        


                print(instruct, number_to_note(note_number), note_number, velocity, timestamp)
                
                if timestamp < 5000:
                    eventArr.append(event)
                #I dont think we use these
                noteArr.append(note_number)
                velocityArr.append(velocity)
                timestampArr.append(timestamp)

            midi_evs = pygame.midi.midis2events(
                midi_events, input_device.device_id)

            for m_e in midi_evs:
                event_post(m_e)

        pygame.time.wait(10)


    # create a midi file
    # create your MIDI object


    #MIDO implementation
    mid = mido.MidiFile()
    m_track = mido.MidiTrack()
    mid.tracks.append(m_track)
    m_track.append(mido.Message('program_change', program=0, time=0))
    m_track.append(mido.MetaMessage('set_tempo', tempo = mido.bpm2tempo(120)))
    m_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))


    # add some notes
    #channel = 0
    volume = 100
    time_counter = 0
    #print("ticksbeat",mid.ticks_per_beat)
    '''
    MIDO default ticks per quarter note = 480
    default tempo / time sig is 120 , 4/4
    default microseconds / quarternote = 500 000
    microseconds/tick  = microseconds/qn / ticks/qn
    therefore microseconds/tick = 500000/480 
    '''
    us_per_tick = 500000/mid.ticks_per_beat
    #TODO EDGE case , note sustained while timer runs out

    m_track.append(mido.Message('note_on', note=0, velocity=1, time= 0)) #dummy event for timing
    for event in eventArr:
        m_time = event[1]*1000 #time from pygame is in seconds, convert to micro
        pitch = event[0][1]
        delta_t = m_time - time_counter
        #print(m_time/1000, time_counter/1000, delta_t/1000 )
        time_counter = m_time
        ticks = int(delta_t/us_per_tick)
        instruct = event[0][0] 

        #print ("ticks:", ticks)
        if instruct == 144:

            m_track.append(mido.Message('note_on', note=pitch, velocity=volume, time= ticks))
        else: 
            m_track.append(mido.Message('note_off', note=pitch, velocity=volume, time= ticks))

    end_ticks = (5*1000*1000)/us_per_tick
    dummy_ticks = int(end_ticks - time_counter/us_per_tick)
    #print("remainder:", (dummy_ticks*us_per_tick)/(1000*1000))
    m_track.append(mido.Message('note_off', note=0, velocity=1, time= dummy_ticks)) #dummy event for timing
    
    m_track.append(mido.MetaMessage('end_of_track'))
    #deal with any note_ons that never turned off before timer ended
    '''
    for pend in pendingArr:
        fakeevent = copy.deepcopy(pend)
        fakeevent[0][0] = 128
        fakeevent[1] = 5000
        pairArr.append((pend, fakeevent))
    print("Read complete")
    '''
    '''
    for pair in pairArr:
        print (pair)
    
    for pair in pairArr:
        note_on = pair[0]
        note_off = pair[1]
        time_on = note_on[1]*1000       #this time is in real time in microseconds
        time_off = note_off[1]*1000     #time in microseconds
        pitch = note_on[0][1]          

        delta_t_on = time_on - time_counter
        time_counter  = time_off
        delta_t_off = time_off - time_on
        
        ticks_on = int(delta_t_on/ms_per_tick)
        ticks_off = int(delta_t_off/ms_per_tick)

        #duration = (time_off - time_on)/seconds_per_beat        
        #mf.addNote(track, channel, pitch, m_time, duration, volume)

        #MIDO
        m_track.append(mido.Message('note_on', note=pitch, velocity=volume, time= ticks_on))
        m_track.append(mido.Message('note_off', note=pitch, velocity=volume, time= ticks_off))
    '''


    mid.save("recorded_melody.mid")


    input_device.close()
    pygame.midi.quit()
    pygame.quit()
    


def number_to_note(number):
    notes = ['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']
    return notes[number % 12]


def play_music(music_file):
    pygame.mixer.music.load(music_file)
    clock = pygame.time.Clock()
    pygame.mixer.music.play()
    # check if playback has finished
    while pygame.mixer.music.get_busy():
        clock.tick(30)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS






def convert_from_midi_to_input( begin_note=48,end_note=73, tempo = 120):
  #TAKE DATA ARRAY AND CHANGE THE NOTE RANGE

    note_range=end_note-begin_note
    midi_title = "./recorded_melody.mid"
    track_num = 0

    ####### convert input to monophonic , also takes note of when no notes are on
    pproll = pypianoroll.Multitrack(tempo = tempo)
    pproll.parse_midi(midi_title)
    pproll.write("intermediate")

    #pproll = pypianoroll.parse(midi_title)
    #pypianoroll.write(pproll, "test") 



    track = pproll.tracks[track_num].pianoroll
    duration=track.shape[0]

    prev_note = None
    prev_timenote = None
    for timestep in range(duration): #iterate over all timesteps
        dupe_chord_flag = 0
        timenote=[]

        for note in range(1,127): #count which notes are being played at this timestep
            velocity=track[timestep][note]
            if velocity !=0:              
                timenote.append(note)
        if (timestep!=0):
            if (np.array_equal(prev_timenote , timenote)) and len(timenote)>1:         
                dupe_chord_flag = 1
        if (len(timenote)>1): #if we have a chord
            #if its the same as the previous chord, use the note we used previously, otherwise randomly sample one
            if dupe_chord_flag == 0: 
                new_note=random.randint(0,len(timenote)-1)
                prev_note = new_note
                prev_timenote = timenote
            else:
                new_note = prev_note
            for i in range(len(timenote)):
                if i != new_note:
                    track[timestep][timenote[i]]=0

        pproll.tracks[track_num].pianoroll[timestep]=track[timestep]

        if (sum(pproll.tracks[track_num].pianoroll[timestep])==0): #this part sets index 0 to = 1 if no notes are being played
            pproll.tracks[track_num].pianoroll[timestep][0] = 1
        
     #######
    pypianoroll.write(pproll, "transformed_melody") 
        
    time_steps= pproll.tracks[track_num].pianoroll.shape[0]
    input_matrix=np.zeros([time_steps,note_range+1])


    for time_step in range(time_steps):

        input_matrix[time_step][0]=((pproll.tracks[track_num].pianoroll)[time_step][0])
        input_matrix[time_step][1:note_range+1]=((pproll.tracks[track_num].pianoroll)[time_step][begin_note:end_note])



    return input_matrix
'''
def forward_pass(input, model):
  #format the input
  
  full_input =  torch.tensor(input).unsqueeze(0).float()[:, :-1]
  song_length = input.shape[0]
  model = model
  full_input = full_input
  output, hidden = model(full_input)

  output, hiddden = output[0][-1], hidden[-1] #get the hidden state and the final note that reflect all of the input

  pred_list = []
  for i in range(0, song_length): #loop for the same time length as input to get cooresponding output notes by passing in one note at a time
    pred = torch.argmax(output) #get the prediction
    pred_list.append(int(pred))
    input = torch.zeros([1, 1, output.shape[0]]) #create a new one hot tensor as our next input note
    input[0][0][int(pred)] = 1
    output, hidden = model(input, hidden) #repeat what we did
    output, hiddden = output[0][-1], hidden[-1]
  #print(pred_list)
  return pred_list
'''

def forward_pass(input, model, alter_output = True, granularity = 6):
  #format the input
  full_input =  torch.tensor(input).unsqueeze(0).float()[:, :-1]
  #print("input",full_input[0], full_input.size())
  song_length = input.shape[0]

  output, hidden = model(full_input)

  output, hiddden = output[0][-1], hidden[-1] #get the hidden state and the final note that reflect all of the input
  input_list=[]
  for note in range(song_length-1):
    input_list.append(int(torch.argmax(full_input[0][note])))
  #print("input_list", input_list)
  #print(output.shape)


  pred_list = []
  for i in range(0, song_length): #loop for the same time length as input to get cooresponding output notes by passing in one note at a time
    pred = torch.argmax(output) #get the prediction
    pred_list.append(int(pred))
    input = torch.zeros([1, 1, output.shape[0]]) #create a new one hot tensor as our next input note
    input[0][0][int(pred)] = 1
    output, hidden = model(input, hidden) #repeat what we did
    output, hiddden = output[0][-1], hidden[-1]
  #print("pred_list",pred_list)

  

  if alter_output == True:
    total_notes = len(pred_list)
    altered_pred_list = []
    for i in range(0, total_notes): #initialize empty list of notes
        altered_pred_list.append(None) 
    note_index = 0
    while note_index < total_notes -1:
        try:
            note = pred_list[note_index]
            if (note != 0):
                for i in range(0, granularity):
                    altered_pred_list[note_index] = note
                    note_index = note_index + 1
            else:
                altered_pred_list[note_index] = 0
                note_index = note_index + 1
        except:
            #print("except", note_index)
            print(" ")
    #print("newpreds:", len(altered_pred_list))
    return altered_pred_list  

  else:
    return pred_list
  '''
  for notes in range(len(pred_list)-1):
    if (notes%4==0):
      note1=pred_list[notes]
      first_note=notes
    elif (notes%4==1):
      note2=pred_list[notes]
    elif (notes%4==2):
      note3=pred_list[notes]
    elif (notes%4==3):
      note4=pred_list[notes]
      last_note=notes
 
    if (notes>3 and note1==note3 and note2!= note1 and note2!=note3 and note4==note2):
      print("first",first_note)
      print("last",last_note)
      for i in range(notes-3,notes+1):
        pred_list[i]=note2
      print("repated notes #",notes,"notes", note1,note2, note3,note4)
  print("pred_list2",pred_list)
  '''
  return pred_list

#for i in range(0, 20):
 # print("song",i)
  #forward_pass (data_pairs[i][0], model)


def display_output(input, model, note_range_shift=48, tempoVal=120.0, velocity=100, trackName='test1', granularity = 3, alter_output = True):
  #pass the input through the model
  preds = forward_pass(input, model, granularity = granularity,alter_output = alter_output)

  pred = torch.tensor(preds) 

  #initialize pianoroll with same number of time steps as prediction tensor
  pianoroll = np.zeros((len(pred), 128))


  #loop through prediction matrix and convert to pianoroll
  for i in range(len(pred)):
        pred_i = pred[i]

        #if 0 we dont want to shift by the note range since its silent
        if(pred[i] != 0):
            pred_i = pred[i] + note_range_shift
            pianoroll[i, [pred_i]]= velocity


  # Create a `pypianoroll.Track` instance
  track = Track(pianoroll=pianoroll, program=0, is_drum=False,
                name=trackName)

  # Create a `pypianoroll.Multitrack` instance
  multitrack = Multitrack(tracks=[track], tempo=tempoVal)


  #convert pianoroll into midi file and write to directory
  if alter_output == True:
    multitrack.write('melody_extension_altered.mid')
  else:    
    multitrack.write('melody_extension.mid')

if __name__ == "__main__":
    # app.run(debug=True)
    countdown(3)

    pygame.init()
    # # Initialization parameters for the mixer
    # freq = 44100    # audio CD quality
    # bitsize = -16   # unsigned 16 bit
    # channels = 2    # 1 is mono, 2 is stereo
    # buffer = 1024   # number of samples

    # # initialize mixer
    # pygame.mixer.init(freq, bitsize, channels, buffer)

    # # Volume 0 to 1.0
    # pygame.mixer.music.set_volume(1.0)

    # # Plat the output of the model
    # try:
    #     music_file = "out.midi"
    #     play_music(music_file)
    # except KeyboardInterrupt:
    #     pygame.mixer.music.fadeout(1000)
    #     pygame.mixer.music.stop()
    #     raise SystemExit

    pygame.midi.init()
    #print_devices()

    # replace id field with id of piano printed
    
    my_input = pygame.midi.Input(1)  # only in my case the id is 2
    my_input = pygame.midi.get_default_input_id()
    input_device = pygame.midi.Input(my_input)
    
    readInput(input_device)
    start_time = time.time()
    
    net = SingleNoteRNN(hidden_size=256)
    state = torch.load("model", map_location = "cpu")
    net.load_state_dict(state)
    
    input_matrix = convert_from_midi_to_input()
    display_output( input_matrix, net, granularity = 6, alter_output = True)
    processing_time = time.time() - start_time
    print("Processing time:", processing_time)
    display_output( input_matrix, net, alter_output = False)


    #verification of song lengths
    test_file = mido.MidiFile("./recorded_melody.mid")
    print("recorded length:", test_file.length)
    test_file = mido.MidiFile("./intermediate.mid")
    print("test length:", test_file.length)
    test_file = mido.MidiFile("./transformed_melody.mid")
    print("transformed length:", test_file.length)
    test_file = mido.MidiFile("./melody_extension.mid")
    print("extension length:", test_file.length)
    test_file = mido.MidiFile("./melody_extension_altered.mid")
    print("altered extension length:", test_file.length)
    # # Initialization parameters for the mixer
    freq = 44100    # audio CD quality
    bitsize = -16   # unsigned 16 bit
    channels = 2    # 1 is mono, 2 is stereo
    buffer = 1024   # number of samples

    # # initialize mixer
    pygame.mixer.init(freq, bitsize, channels, buffer)

    # # Volume 0 to 1.0
    pygame.mixer.music.set_volume(1.0)
    try:
        print("What you played")
        music_file = "transformed_melody.mid"
        play_music(music_file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
    try:
        print("What was generated")
        music_file = "melody_extension.mid"
        play_music(music_file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit
    try:
        print("What was generated and altered")
        music_file = "melody_extension_altered.mid"
        play_music(music_file)
    except KeyboardInterrupt:
        pygame.mixer.music.fadeout(1000)
        pygame.mixer.music.stop()
        raise SystemExit

