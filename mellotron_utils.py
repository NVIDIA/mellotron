import re
import numpy as np
import music21 as m21
import torch
import torch.nn.functional as F
from text import text_to_sequence, get_arpabet, cmudict


CMUDICT_PATH = "data/cmu_dictionary"
CMUDICT = cmudict.CMUDict(CMUDICT_PATH)
PHONEME2GRAPHEME = {
    'AA': ['a', 'o', 'ah'],
    'AE': ['a', 'e'],
    'AH': ['u', 'e', 'a', 'h', 'o'],
    'AO': ['o', 'u', 'au'],
    'AW': ['ou', 'ow'],
    'AX': ['a'],
    'AXR': ['er'],
    'AY': ['i'],
    'EH': ['e', 'ae'],
    'EY': ['a', 'ai', 'ei', 'e', 'y'],
    'IH': ['i', 'e', 'y'],
    'IX': ['e', 'i'],
    'IY': ['ea', 'ey', 'y', 'i'],
    'OW': ['oa', 'o'],
    'OY': ['oy'],
    'UH': ['oo'],
    'UW': ['oo', 'u', 'o'],
    'UX': ['u'],
    'B': ['b'],
    'CH': ['ch', 'tch'],
    'D': ['d', 'e', 'de'],
    'DH': ['th'],
    'DX': ['tt'],
    'EL': ['le'],
    'EM': ['m'],
    'EN': ['on'],
    'ER': ['i', 'er'],
    'F': ['f'],
    'G': ['g'],
    'HH': ['h'],
    'JH': ['j'],
    'K': ['k', 'c', 'ch'],
    'KS': ['x'],
    'L': ['ll', 'l'],
    'M': ['m'],
    'N': ['n', 'gn'],
    'NG': ['ng'],
    'NX': ['nn'],
    'P': ['p'],
    'Q': ['-'],
    'R': ['wr', 'r'],
    'S': ['s', 'ce'],
    'SH': ['sh'],
    'T': ['t'],
    'TH': ['th'],
    'V': ['v', 'f', 'e'],
    'W': ['w'],
    'WH': ['wh'],
    'Y': ['y', 'j'],
    'Z': ['z', 's'],
    'ZH': ['s']
}

########################
#  CONSONANT DURATION  #
########################
PHONEMEDURATION = {
    'B': 0.05,
    'CH': 0.1,
    'D': 0.075,
    'DH': 0.05,
    'DX': 0.05,
    'EL': 0.05,
    'EM': 0.05,
    'EN': 0.05,
    'F': 0.1,
    'G': 0.05,
    'HH': 0.05,
    'JH': 0.05,
    'K': 0.05,
    'L': 0.05,
    'M': 0.15,
    'N': 0.15,
    'NG': 0.15,
    'NX': 0.05,
    'P': 0.05,
    'Q': 0.075,
    'R': 0.05,
    'S': 0.1,
    'SH': 0.05,
    'T': 0.075,
    'TH': 0.1,
    'V': 0.05,
    'Y': 0.05,
    'W': 0.05,
    'WH': 0.05,
    'Z': 0.05,
    'ZH': 0.05
}


def add_space_between_events(events, connect=False):
    new_events = []
    for i in range(1, len(events)):
        token_a, freq_a, start_time_a, end_time_a = events[i-1][-1]
        token_b, freq_b, start_time_b, end_time_b = events[i][0]

        if token_a in (' ', '') and len(events[i-1]) == 1:
            new_events.append(events[i-1])
        elif token_a not in (' ', '') and token_b not in (' ', ''):
            new_events.append(events[i-1])
            if connect:
                new_events.append([[' ', 0, end_time_a, start_time_b]])
            else:
                new_events.append([[' ', 0, end_time_a, end_time_a]])
        else:
            new_events.append(events[i-1])

    if new_events[-1][0][0] != ' ':
        new_events.append([[' ', 0, end_time_a, end_time_a]])
    new_events.append(events[-1])

    return new_events


def adjust_words(events):
    new_events = []
    for event in events:
        if len(event) == 1 and event[0][0] == ' ':
            new_events.append(event)
        else:
            for e in event:
                if e[0][0].isupper():
                    new_events.append([e])
                else:
                    new_events[-1].extend([e])
    return new_events


def adjust_extensions(events, phoneme_durations):
    if len(events) == 1:
        return events

    idx_last_vowel = None
    n_consonants_after_last_vowel = 0
    target_ids = np.arange(len(events))
    for i in range(len(events)):
        token = re.sub('[0-9{}]', '', events[i][0])
        if idx_last_vowel is None and token not in phoneme_durations:
            idx_last_vowel = i
            n_consonants_after_last_vowel = 0
        else:
            if token == '_' and not n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
            elif token == '_' and n_consonants_after_last_vowel:
                events[i][0] = events[idx_last_vowel][0]
                start = idx_last_vowel + 1
                target_ids[start:start+n_consonants_after_last_vowel] += 1
                target_ids[i] -= n_consonants_after_last_vowel
            elif token in phoneme_durations:
                n_consonants_after_last_vowel += 1
            else:
                n_consonants_after_last_vowel = 0
                idx_last_vowel = i

    new_events = [0] * len(events)
    for i in range(len(events)):
        new_events[target_ids[i]] = events[i]

    # adjust time of consonants that were repositioned
    for i in range(1, len(new_events)):
        if new_events[i][2] < new_events[i-1][2]:
            new_events[i][2] = new_events[i-1][2]
            new_events[i][3] = new_events[i-1][3]

    return new_events


def adjust_consonant_lengths(events, phoneme_durations):
    t_init = events[0][2]

    idx_last_vowel = None
    for i in range(len(events)):
        task = re.sub('[0-9{}]', '', events[i][0])
        if task in phoneme_durations:
            duration = phoneme_durations[task]
            if idx_last_vowel is None:  # consonant comes before any vowel
                events[i][2] = t_init
                events[i][3] = t_init + duration
            else:  # consonant comes after a vowel, must offset
                events[idx_last_vowel][3] -= duration
                for k in range(idx_last_vowel+1, i):
                    events[k][2] -= duration
                    events[k][3] -= duration
                events[i][2] = events[i-1][3]
                events[i][3] = events[i-1][3] + duration
        else:
            events[i][2] = t_init
            events[i][3] = events[i][3]
            t_init = events[i][3]
            idx_last_vowel = i
        t_init = events[i][3]

    return events


def adjust_consonants(events, phoneme_durations):
    if len(events) == 1:
        return events

    start = 0
    split_ids = []
    t_init = events[0][2]

    # get each substring group
    for i in range(1, len(events)):
        if events[i][2] != t_init:
            split_ids.append((start, i))
            start = i
            t_init = events[i][2]
    split_ids.append((start, len(events)))

    for (start, end) in split_ids:
        events[start:end] = adjust_consonant_lengths(
            events[start:end], phoneme_durations)

    return events


def adjust_event(event, hop_length=256, sampling_rate=22050):
    tokens, freq, start_time, end_time = event

    if tokens == ' ':
        return [event] if freq == 0 else [['_', freq, start_time, end_time]]

    return [[token, freq, start_time, end_time] for token in tokens]


def musicxml2score(filepath, bpm=60):
    track = {}
    beat_length_seconds = 60/bpm
    data = m21.converter.parse(filepath)
    for i in range(len(data.parts)):
        part = data.parts[i].flat
        events = []
        for k in range(len(part.notesAndRests)):
            event = part.notesAndRests[k]
            if isinstance(event, m21.note.Note):
                freq = event.pitch.frequency
                token = event.lyrics[0].text if len(event.lyrics) > 0 else ' '
                start_time = event.offset * beat_length_seconds
                end_time = start_time + event.duration.quarterLength * beat_length_seconds
                event = [token, freq, start_time, end_time]
            elif isinstance(event, m21.note.Rest):
                freq = 0
                token = ' '
                start_time = event.offset * beat_length_seconds
                end_time = start_time + event.duration.quarterLength * beat_length_seconds
                event = [token, freq, start_time, end_time]

            if token == '_':
                raise Exception("Unexpected token {}".format(token))

            if len(events) == 0:
                events.append(event)
            else:
                if token == ' ':
                    if freq == 0:
                        if events[-1][1] == 0:
                            events[-1][3] = end_time
                        else:
                            events.append(event)
                    elif freq == events[-1][1]:  # is event duration extension ?
                        events[-1][-1] = end_time
                    else:  # must be different note on same syllable
                        events.append(event)
                else:
                    events.append(event)
        track[part.partName] = events
    return track


def track2events(track):
    events = []
    for e in track:
        events.extend(adjust_event(e))
    group_ids = [i for i in range(len(events))
                 if events[i][0] in [' '] or events[i][0].isupper()]

    events_grouped = []
    for i in range(1, len(group_ids)):
        start, end = group_ids[i-1], group_ids[i]
        events_grouped.append(events[start:end])

    if events[-1][0] != ' ':
        events_grouped.append(events[group_ids[-1]:])

    return events_grouped


def events2eventsarpabet(event):
    if event[0][0] == ' ':
        return event

    # get word and word arpabet
    word = ''.join([e[0] for e in event if e[0] not in('_', ' ')])
    word_arpabet = get_arpabet(word, CMUDICT)
    if word_arpabet[0] != '{':
        return event

    word_arpabet = word_arpabet.split()

    # align tokens to arpabet
    i, k = 0, 0
    new_events = []
    while i < len(event) and k < len(word_arpabet):
        # single token
        token_a, freq_a, start_time_a, end_time_a = event[i]

        if token_a == ' ':
            new_events.append([token_a, freq_a, start_time_a, end_time_a])
            i += 1
            continue

        if token_a == '_':
            new_events.append([token_a, freq_a, start_time_a, end_time_a])
            i += 1
            continue

        # two tokens
        if i < len(event) - 1:
            j = i + 1
            token_b, freq_b, start_time_b, end_time_b = event[j]
            between_events = []
            while j < len(event) and event[j][0] == '_':
                between_events.append([token_b, freq_b, start_time_b, end_time_b])
                j += 1
                if j < len(event):
                    token_b, freq_b, start_time_b, end_time_b = event[j]

            token_compound_2 = (token_a + token_b).lower()

        # single arpabet
        arpabet = re.sub('[0-9{}]', '', word_arpabet[k])

        if k < len(word_arpabet) - 1:
            arpabet_compound_2 = ''.join(word_arpabet[k:k+2])
            arpabet_compound_2 = re.sub('[0-9{}]', '', arpabet_compound_2)

        if i < len(event) - 1 and token_compound_2 in PHONEME2GRAPHEME[arpabet]:
            new_events.append([word_arpabet[k], freq_a, start_time_a, end_time_a])
            if len(between_events):
                new_events.extend(between_events)
            if start_time_a != start_time_b:
                new_events.append([word_arpabet[k], freq_b, start_time_b, end_time_b])
            i += 2 + len(between_events)
            k += 1
        elif token_a.lower() in PHONEME2GRAPHEME[arpabet]:
            new_events.append([word_arpabet[k], freq_a, start_time_a, end_time_a])
            i += 1
            k += 1
        elif arpabet_compound_2 in PHONEME2GRAPHEME and token_a.lower() in PHONEME2GRAPHEME[arpabet_compound_2]:
            new_events.append([word_arpabet[k], freq_a, start_time_a, end_time_a])
            new_events.append([word_arpabet[k+1], freq_a, start_time_a, end_time_a])
            i += 1
            k += 2
        else:
            k += 1

    # add extensions and pauses at end of words
    while i < len(event):
        token_a, freq_a, start_time_a, end_time_a = event[i]

        if token_a in (' ', '_'):
            new_events.append([token_a, freq_a, start_time_a, end_time_a])
        i += 1

    return new_events


def event2alignment(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)

    n_frames = int(events[-1][-1][-1] / frame_length)
    n_tokens = np.sum([len(e) for e in events])
    alignment = np.zeros((n_tokens, n_frames))

    cur_event = -1
    for event in events:
        for i in range(len(event)):
            if len(event) == 1 or cur_event == -1 or event[i][0] != event[i-1][0]:
                cur_event += 1
            token, freq, start_time, end_time = event[i]
            alignment[cur_event, int(start_time/frame_length):int(end_time/frame_length)] = 1

    return alignment[:cur_event+1]


def event2f0(events, hop_length=256, sampling_rate=22050):
    frame_length = float(hop_length) / float(sampling_rate)
    n_frames = int(events[-1][-1][-1] / frame_length)
    f0s = np.zeros((1, n_frames))

    for event in events:
        for i in range(len(event)):
            token, freq, start_time, end_time = event[i]
            f0s[0, int(start_time/frame_length):int(end_time/frame_length)] = freq

    return f0s


def event2text(events, convert_stress, cmudict=None):
    text_clean = ''
    for event in events:
        for i in range(len(event)):
            if i > 0 and event[i][0] == event[i-1][0]:
                continue
            if event[i][0] == ' ' and len(event) > 1:
                if text_clean[-1] != "}":
                    text_clean = text_clean[:-1] + '} {'
                else:
                    text_clean += ' {'
            else:
                if event[i][0][-1] in ('}', ' '):
                    text_clean += event[i][0]
                else:
                    text_clean += event[i][0] + ' '

    if convert_stress:
        text_clean = re.sub('[0-9]', '1', text_clean)

    text_encoded = text_to_sequence(text_clean, [], cmudict)
    return text_encoded, text_clean


def remove_excess_frames(alignment, f0s):
    excess_frames = np.sum(alignment.sum(0) == 0)
    alignment = alignment[:, :-excess_frames] if excess_frames > 0 else alignment
    f0s = f0s[:, :-excess_frames] if excess_frames > 0 else f0s
    return alignment, f0s


def get_data_from_musicxml(filepath, bpm, phoneme_durations=None,
                           convert_stress=False):
    if phoneme_durations is None:
        phoneme_durations = PHONEMEDURATION
    score = musicxml2score(filepath, bpm)
    data = {}
    for k, v in score.items():
        # ignore empty tracks
        if len(v) == 1 and v[0][0] == ' ':
            continue

        events = track2events(v)
        events = adjust_words(events)
        events_arpabet = [events2eventsarpabet(e) for e in events]

        # make adjustments
        events_arpabet = [adjust_extensions(e, phoneme_durations)
                          for e in events_arpabet]
        events_arpabet = [adjust_consonants(e, phoneme_durations)
                          for e in events_arpabet]
        events_arpabet = add_space_between_events(events_arpabet)

        # convert data to alignment, f0 and text encoded
        alignment = event2alignment(events_arpabet)
        f0s = event2f0(events_arpabet)
        alignment, f0s = remove_excess_frames(alignment, f0s)
        text_encoded, text_clean = event2text(events_arpabet, convert_stress)

        # convert data to torch
        alignment = torch.from_numpy(alignment).permute(1, 0)[:, None].float()
        f0s = torch.from_numpy(f0s)[None].float()
        text_encoded = torch.LongTensor(text_encoded)[None]
        data[k] = {'rhythm': alignment,
                   'pitch_contour': f0s,
                   'text_encoded': text_encoded}

    return data


if __name__ == "__main__":
    import argparse
    # Get defaults so it can work with no Sacred
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filepath", required=True)
    args = parser.parse_args()
    get_data_from_musicxml(args.filepath, 60)
