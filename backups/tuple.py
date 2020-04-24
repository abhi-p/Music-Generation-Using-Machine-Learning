'''
list entire directory
get unique file #s ->___._.midi
load pairs of files into lists of tuples
'''

max_counter=100
counter=0

#contains an array of tuples(input, target)
tupleArr=[]

#contains a 2 element tuple, converted into tuple type later
tuple_list=[]

dir_path = "drive/My Drive/Capstone - ECE496/single_note_random_split"
dir_files = os.listdir(dir_path)
dir_files.sort()
current_song=""

for midi_file_num in dir_files:
  #set flag to false at beginning of each iteration
  new_song = False

  #checks if we are in a new song or current song
  if(current_song != midi_file_num.split(".")[0]):
    new_song = True
    current_song = midi_file_num.split(".")[0]

  #gets the current song interval part  
  current_part = midi_file_num.split(".")[1]

  #if in a new song, reset variables
  if new_song:
    index=0
    tuple_list=[]

  #add element to tuple
  tuple_list.append(mido.MidiFile(dir_path +"/" + midi_file_num))

  #if tuple(input, target) complete, add to array of tuples, reset vars
  if(index != 1):
    index=index+1
  elif(index ==1):

    #tuple has both input and target now
    t = tuple(tuple_list)
    #print(t)
    tupleArr.append(t)

    #reset and add initial element to tuple
    tuple_list = []
    tuple_list.append(mido.MidiFile(dir_path +"/" + midi_file_num))

  if(counter == max_counter):
    break
  counter = counter+1
