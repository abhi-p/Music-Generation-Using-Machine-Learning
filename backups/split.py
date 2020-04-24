#splitting files by time
#TODO change int casts to rounding for tempo and time


input_dir = drive_path+'/single_note_random/'
output_dir = drive_path+'/single_note_random_split/'

drive.mount('/content/drive', force_remount= True)


def split_midi_by_time (midi_title, interval_len):
  file_number = midi_title.split('.')[0]
  pproll = pypianoroll.Multitrack()
  pproll.parse_midi(input_dir + midi_title) 
  print(midi_title)
    



  #find the tempo message and extract the tempo of the song (needed to recreate the song)
  song_tempo = 0
  test_file = mido.MidiFile(input_dir + midi_title)
  for msg in test_file:
    if msg.is_meta:
        split = str(msg).split()
        if 'set_tempo' in split:
            song_tempo = int(mido.tempo2bpm(int(split[3].split('=')[1]))) #mido uses tempo in ms/beat but pypianoroll uses beat/min as tempo
            break
  #print("tempo ", song_tempo)

  song_len = int(test_file.length) 
  #print(song_len , "seconds")

  num_steps = len(pproll.tracks[0].pianoroll)
  #print('total steps', num_steps)
  time_steps_per_second = int(num_steps/song_len)
  #print("steps per second", time_steps_per_second)

  num_intervals = song_len/interval_len
  #print('total intervals', num_intervals)

  steps_per_interval = time_steps_per_second * interval_len
  #print(steps_per_interval)

  for i in range(int(num_intervals) -1):
      tracks = []
      title = file_number + "." + str(i) + ".midi"
      
      tracks.append( Track(pianoroll = pproll.tracks[0].pianoroll[i*steps_per_interval:(i+1)*steps_per_interval] ) )
      tracks.append( Track(pianoroll = pproll.tracks[1].pianoroll[i*steps_per_interval:(i+1)*steps_per_interval] ) )
      new_pproll = pypianoroll.Multitrack(tracks = tracks, tempo = song_tempo )
      pypianoroll.write(new_pproll,  output_dir +title)
      #new_pproll.plot()



