import py_midicsv as pm
import sys

scale_index_to_string = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
global_transform = -29


def convert_to_csv(filename, transpose=0, min_interval=96, meta_filepath='meta.csv', voice=0):
    midi = pm.midi_to_csv(filename)

    index = 0
    meta_text = ''
    # Index of first note in the midi, which we assume is the upper voice
    while index < len(midi):
        itemized = midi[index].split(',')
        index += 1
        if len(itemized) > 2 and itemized[2] == ' Note_on_c':
            next_event_index_upper = index - 1
            break
        else:
            meta_text += midi[index - 1]

    meta_data = open(meta_filepath, 'w')
    meta_data.write(meta_text[:-1])
    meta_data.close()

    # Index to the end of this midi track
    while index < len(midi):
        itemized = midi[index].split(',')
        index += 1
        if len(itemized) > 2 and itemized[2][:10] == ' End_track':
            break

    # Index of the first note in the next track, which we assume is the lower track
    while index < len(midi):
        itemized = midi[index].split(',')
        if len(itemized) > 2 and itemized[2] == ' Note_on_c':
            next_event_index_lower = index
            break
        index += 1

    # Rest = 1, hold = 0
    current_event_upper = '1'
    current_event_lower = '1'

    time = -min_interval  # Start here, so our first loop puts us at 0
    csv = ''
    # while next_event_time_upper != -1 and next_event_time_lower != -1:

    # Loop: advance time, and apply effects of any events at the current time
    while time <= 1000000:
        time += min_interval
        if voice != 2:
            upper_note_itemized = midi[next_event_index_upper].split(',')
            # For every event at this position in time (and any we just jumped over in time)
            while int(upper_note_itemized[1]) <= time:
                # Update the current event, and advance the anticipated next note by one
                if upper_note_itemized[2][:10] == ' End_track':
                    time = 1000001
                    break
                elif upper_note_itemized[2] == ' Note_off_c':
                    current_event_upper = '1'
                elif upper_note_itemized[2] == ' Note_on_c':
                    current_event_upper = str(int(upper_note_itemized[4][1:]) + transpose)
                next_event_index_upper += 1
                upper_note_itemized = midi[next_event_index_upper].split(',')
            csv += current_event_upper
            if voice == 1:
                csv += '\n'
            else:
                csv += ','
            current_event_upper = '0'

        if voice != 1:
            lower_note_itemized = midi[next_event_index_lower].split(',')
            # For every event at this position in time (and any we just jumped over in time)
            while int(lower_note_itemized[1]) <= time:
                # Update the current event, and advance the anticipated next note by one
                if lower_note_itemized[2][:10] == ' End_track':
                    time = 1000001
                    break
                elif lower_note_itemized[2] == ' Note_off_c':
                    current_event_lower = '1'
                elif lower_note_itemized[2] == ' Note_on_c':
                    current_event_lower = str(int(lower_note_itemized[4][1:]) + transpose)
                next_event_index_lower += 1
                lower_note_itemized = midi[next_event_index_lower].split(',')
            csv += current_event_lower + '\n'
            current_event_lower = '0'

    file = open('transcription.csv', 'w')
    file.write(csv)
    file.close()
    print('output complete')

    return csv


def convert_to_midi(transcription_path='transcription.csv', metadata_path='meta.csv', output_path='midi_out.mid',
                    transpose=0, min_interval=96, tempo=None):
    min_interval = 96
    metadata_file = open(metadata_path, 'r')
    metadata = metadata_file.read().split('\n')
    metadata_file.close()

    transcription = open(transcription_path, 'r')
    trans_list = transcription.read().split('\n')
    transcription.close()

    midi_csv_list = []
    for i in range(0, len(metadata)):
        if metadata[1] != '':
            midi_csv_list.append(metadata[i] + '\n')
    time = 0
    prev_pitch = 0
    for i in range(1, len(trans_list)):
        itemized = trans_list[i].split(',')
        if len(itemized) == 0:
            break
        if itemized[0] == '1':
            if prev_pitch != 0:
                midi_csv_list.append('2, ' + str(time) + ', Note_off_c, 1, ' + str(prev_pitch) + ', 64\n')
                prev_pitch = 0
        elif itemized[0] != '0':
            if prev_pitch != 0:
                midi_csv_list.append('2, ' + str(time) + ', Note_off_c, 1, ' + str(prev_pitch) + ', 64\n')
            if itemized[0] != '':
                prev_pitch = str(int(itemized[0]) + transpose - global_transform)
            midi_csv_list.append('2, ' + str(time) + ', Note_on_c, 1, ' + str(prev_pitch) + ', 127\n')
        time += min_interval

    if len(trans_list[0]) == 1:
        midi_csv_list.append('3, ' + str(time) + ', End_track\n')
        midi_csv_list.append('0, 0, End_of_file')
    else:

        midi_csv_list.append('2, ' + str(time) + ', End_track\n')
        midi_csv_list.append('3, 0, Start_track\n')
        midi_csv_list.append('3, 0, Title_t, \"two\"\n')

        time = 0
        prev_pitch = 0
        for i in range(1, len(trans_list)):
            itemized = trans_list[i].split(',')
            if len(itemized) != 2:
                break
            if itemized[1] == '1':
                if prev_pitch != 0:
                    midi_csv_list.append('3, ' + str(time) + ', Note_off_c, 2, ' + str(prev_pitch) + ', 64\n')
                    prev_pitch = 0
            elif itemized[1] != '0':
                if prev_pitch != 0:
                    midi_csv_list.append('3, ' + str(time) + ', Note_off_c, 2, ' + str(prev_pitch) + ', 64\n')
                prev_pitch = str(int(itemized[1]) + transpose - global_transform)
                midi_csv_list.append('3, ' + str(time) + ', Note_on_c, 2, ' + str(prev_pitch) + ', 127\n')
            time += min_interval

        midi_csv_list.append('3, ' + str(time) + ', End_track\n')
        midi_csv_list.append('0, 0, End_of_file')

    midi_csv = ''
    for item in midi_csv_list:
        midi_csv += item
    output_csv = open('midi_readable.csv', 'w')
    output_csv.write(midi_csv)
    output_csv.close()

    midi_obj = pm.csv_to_midi(midi_csv_list)

    midi_out = open(output_path, 'wb')
    writer = pm.FileWriter(midi_out)
    writer.write(midi_obj)


if __name__ == '__main__':

    # Transposition amounts
    # inv 1 = 0
    # inv 3 = -2
    # inv 5 = -3
    # inv 8 = -5
    # inv 10 = -7
    # inv 12 = +3

    out = convert_to_csv(sys.argv[1], int(sys.argv[2]) + global_transform, voice=1)
    for i in range(3, len(sys.argv) - 1, 2):
        out += convert_to_csv(sys.argv[i], sys.argv[i + 1] + global_transform, voice=1)
    out_file = open('combined_out.csv', 'w')
    out_file.write(out)
    out_file.close()
    # convert_to_midi()
