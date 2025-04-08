import numpy as np
from typing import List, Dict, Tuple

import pyquist as pq
import pyquist.helper as pqh

from pyquist.web.theorytab import fetch_theorytab_json, theorytab_json_to_score
from pyquist.score import Score, PlayableScore, render_score, BasicMetronome, Instrument
from pyquist.web.freesound import fetch

import json


def part1_inst(duration: float, pitch: float, sample_rate: int = 44100) -> pq.Audio:
    """Generates a sine wave at the requested pitch and sample_rate,
    for the duration in seconds, with zero phase offset.

    For durations which do not evenly combine with sample_rate, floor the
    resulting product.
    
    Returns: The generated signal.
    """
    # pitch to hz frequency (by random formula on the internet)
    freq = 440.0 * 2 ** ((pitch - 69) / 12.0)  

    # number of samples (flooring in case we have a fraction)
    num_samples = int(np.floor(duration * sample_rate))
    
    # numpy array from 0 to duration 
    t = np.linspace(0.0, duration, num_samples, endpoint=False)
    
    # make sine wave at the calculated frequency 
    sine_wave = np.sin(2 * np.pi * freq * t)
    
    # numpy array to pyquist audio object
    return pq.Audio.from_array(sine_wave, sample_rate=sample_rate)



def part1_score(times: np.ndarray, durations: np.ndarray, pitches: np.ndarray, inst: Instrument) -> PlayableScore:
    """Generates a PlayableScore containing the specified sound events. Times
    and durations are in seconds, pitches is in steps.

    Utilize part1_inst when turning sound events into playable sound events. 
    
    Returns: The constructed score.
    """

    assert len(times) == len(durations) and len(durations) == len(pitches), \
        "Lengths of input arguments did not match"
    
    score = []
    for onset, duration, pitch in zip(times, durations, pitches):
        # make kwargs dictionary for sound
        kwargs = {"duration": duration, "pitch": pitch}
        # add sound (time, instrument, kwargs) to score
        score.append((onset, inst, kwargs))
    return score



class Envelope:
    """a linearly interpolating envelope."""

    def __init__(self, times: np.ndarray, values: np.ndarray):
        if not np.all(np.diff(times) > 0):
            raise ValueError("Times must be strictly increasing.")
        if len(times) != len(values):
            raise ValueError("Times and values must have the same length.")
        if len(times) < 2:
            raise ValueError("Envelope must have at least two points.")
        
        self.times = times
        self.values = values
    
    def __call__(self, t: np.ndarray) -> np.ndarray:
        """
        render specified t values from the piecewise times and values from init - 
        t values outside the range of self.times are set to the nearest endpoint
        """
        # check t strictly increasing
        if not np.all(np.diff(t) > 0):
            raise ValueError("Query times must be sorted and unique.")
        
        # determine segment index for each value in t::
        # from internet: np.searchsorted(..., side='right') gives the insertion 
        # index, so subtract 1 to find which segment we are in.
        idx = np.searchsorted(self.times, t, side='right') - 1
        
        # prepare an output array of the same shape as t
        out = np.zeros_like(t, dtype=float)
        
        # identify which points fall before first segment, within valid segments,
        # or after last segment
        mask_left = idx < 0
        mask_right = idx >= len(self.times) - 1
        mask_mid = ~(mask_left | mask_right)  # in valid range
        
        # left side (before the earliest time)
        out[mask_left] = self.values[0]
        
        # right side (beyond the latest time)
        out[mask_right] = self.values[-1]
        
        # mid range (piecewise linear interpolation)
        j = idx[mask_mid]                  # segment index
        x0 = self.times[j]                 # left time
        x1 = self.times[j + 1]             # right time
        y0 = self.values[j]                # left value
        y1 = self.values[j + 1]            # right value
        x = t[mask_mid]                    # query times in this segment
        
        # Linear interpolation formula:
        # y = y0 + ( (x - x0) * (y1 - y0) / (x1 - x0) )
        out[mask_mid] = y0 + ( (x - x0) * (y1 - y0) / (x1 - x0) )
        
        return out
    


def part2_inst(duration: float, pitch: float, env: Envelope, sample_rate: int = 44100) -> pq.Audio:
    """Generates a sine wave at the requested pitch and sample_rate,
    for the duration in seconds, with zero phase offset, shaped by the provided envelope.

    For durations which do not evenly combine with sample_rate, floor the
    resulting product.
    
    Returns: The generated signal.
    """

    base_audio = part1_inst(duration, pitch, sample_rate=sample_rate)

    # convert to numpy array
    wave = np.array(base_audio)
    
    # make mono
    if wave.ndim > 1:
        wave = wave[:, 0]   

    n_samples = len(wave)
    t = np.linspace(0.0, duration, n_samples, endpoint=False)

    # run envelope
    env_values = env(t)

    # make env_values is also 1D
    env_values = env_values.flatten()

    # multiply through to shape
    shaped_wave = wave * env_values
    return pq.Audio.from_array(shaped_wave, sample_rate=sample_rate)




def part2_score(times: np.ndarray,
                durations: np.ndarray,
                pitches: np.ndarray,
                env: Envelope,
                inst: Instrument) -> PlayableScore:
    """
    Generates a PlayableScore containing the specified sound events.
    Times and durations are in seconds, pitches is in steps.
    Utilize the provided instrument to shape the notes with the given env.
    """

    assert len(times) == len(durations) == len(pitches), \
        "Lengths of input arguments did not match"

    score = []

    # loop over each event and create a sound event
    for t, dur, pit in zip(times, durations, pitches):
        event = (t,     # onset time
                inst,   # instrument
            {
                'duration': dur,
                'pitch': pit,
                'env': env
            }
        )
        score.append(event)
    return score




def get_harmonic_envelopes(audio: pq.Audio, num_harmonics: int = 8, hop_length: int = 512, debug_plot = False) -> List[Envelope]:
    """Returns the harmonic presence in the audio, shape num_harmonics x
    ceil(audio length / hop_length).
    
    Debug_plot parameter controls whether or not to save spectra_*.png and
    envelopes.png to visualize the process.
    """

    assert num_harmonics >= 1, "Must provide at least one harmonic"
    import librosa
    from matplotlib import pyplot as plt
    n_fft = 2048
    y = np.array(audio).T
    sr = audio.sample_rate
    y = y.mean(axis=0)
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    magnitude, _ = librosa.magphase(D)
    magnitude = magnitude[None]
    y = y[None]
    # Using piano range for now
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'), sr=sr
    )
    f0 = f0[0, :]
    assert not np.isnan(f0).all(), "All fundamental freqs were nan"
    assert f0.ndim == 1, "must have 1 dimension for f0"
    f0_median = np.nanmedian(f0) * np.ones_like(f0)

    harmonics = np.array([f0_median * (i + 1) for i in range(num_harmonics)])

    envelopes = np.zeros((num_harmonics, magnitude.shape[-1]))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    for t in range(magnitude.shape[-1]):
        if debug_plot:
            plt.clf()
            plt.plot(magnitude[0, :, t]); plt.ylim(0, 60)
            plt.savefig(f'spectra_{t}.png')
        # keep as 0 amplitude if no fundamental frequency detected
        if np.isnan(f0[t]):
            continue 
        for h in range(num_harmonics):
            harmonic_freq = harmonics[h, t]
            if np.isnan(harmonic_freq) or harmonic_freq > sr // 2:
                continue
            # Find the closest frequency bin
            bin_idx = np.argmin(np.abs(frequencies - harmonic_freq))
            bin_low = np.floor(bin_idx).astype(int) - 1
            bin_high = np.ceil(bin_idx).astype(int) + 1
            envelopes[h, t] = np.mean(magnitude[0, bin_low:bin_high+1, t])

    # Normalize with the largest total amplitude across all harmonics
    envelope_norm = envelopes.sum(axis=0).max()
    envelopes = envelopes/envelope_norm

    frame_times = librosa.frames_to_time(np.arange(envelopes.shape[1]), sr=sr, hop_length=hop_length)
    total_duration = len(audio) / sr
    times = frame_times * (total_duration / frame_times[-1])
    if debug_plot:
        plt.clf()
        for h in range(num_harmonics):
            plt.plot(times, envelopes[h], label=f"f{h}")
        plt.legend()
        plt.savefig("envelopes.png")
    return [Envelope(times, values) for values in envelopes]


def part2_harmonic_inst(duration: float,
                        pitch: float,
                        harmonic_envs: List[Envelope],
                        overall_env: Envelope = None,
                        sample_rate: int = 44100) -> pq.Audio:
    """
    Generates harmonics via sine waves at the requested pitch and sample_rate,
    for the duration in seconds, with zero phase offset, shaped by the provided envelopes.
    The number of harmonics generated matches the length of the envelopes passed in.
    The final audio is then shaped with the optional provided overall_env, as in part2_inst.
    """

    # pitch to base frequency
    freq_base = 440.0 * 2 ** ((pitch - 69.0) / 12.0)

    n_samples = int(np.floor(duration * sample_rate))
    
    # time vector
    t = np.arange(n_samples) / sample_rate

    # init waveform
    total_wave = np.zeros(n_samples, dtype=float)

    # Sum up each harmonic's sine wave modulated by its envelope.
    for i, env in enumerate(harmonic_envs):
        harmonic_freq = (i + 1) * freq_base
        wave = np.sin(2 * np.pi * harmonic_freq * t)
        env_vals = env(t).flatten()  # Ensure envelope values are 1D
        wave *= env_vals
        total_wave += wave

    if overall_env is not None:
        total_wave *= overall_env(t)

    peak = np.max(np.abs(total_wave))
    if peak > 0:
        total_wave = total_wave * (0.4 / peak)

    return pq.Audio.from_array(total_wave, sample_rate=sample_rate)



def part2_harmonic_score(times: np.ndarray,
                durations: np.ndarray,
                pitches: np.ndarray,
                ref_audio: pq.Audio,
                num_harmonics: int = 8) -> PlayableScore:
    """Generates a PlayableScore containing the specified sound events,
    utilizing the timbre from ref_audio. Times and durations are in seconds,
    pitches is in steps.

    Use the provided get_harmonic_envelopes function to obtain time-varying
    envelopes for each harmonic, then additively synthesize each harmonic
    (starting with the fundamental) for each provided pitch, utilizing sine
    waves with zero phase offset.
    
    Returns: The constructed score.
    """

    assert num_harmonics >= 1, "Must request at least one harmonic"
    assert len(times) == len(durations) and len(durations) == len(pitches), \
        "Lengths of input arguments did not match"

    
    # get harmonic envelopes from reference audio
    harmonic_envs = get_harmonic_envelopes(ref_audio, num_harmonics=num_harmonics, hop_length=512)

    score = []

    # same as part 1 stuff
    for t, dur, pit in zip(times, durations, pitches):
        event = (t, part2_harmonic_inst,
                {
                    'duration': dur,
                    'pitch': pit,
                    'harmonic_envs': harmonic_envs,
                    'overall_env': None
                }
        )
        score.append(event)
    return score


def part3_cover(melody: Score,
                harmony: Score,
                melody_inst: Instrument,
                harmony_inst: Instrument,
                rolled: float = 0) -> PlayableScore:
    """Combined the provided melody and harmony scores into a single, sorted,
    PlayableScore. Chords occuring in harmony are converted from block to rolled
    chords, with each note therein being cummulatively delayed from low to high
    pitches by the rolled parameter.

    Use the provided melody and harmony inst parameters to create playable sound
    events, being careful not to override kwargs when passing along to the
    instruments.
    
    Returns: The constructed score.
    """

    from collections import defaultdict


    assert rolled >= 0, "Rolled duration must be non-negative"



    new_melody = []
    for event in melody:
        if len(event) == 3:
            onset, _, kwargs = event
        elif len(event) == 2:
            onset, kwargs = event
        else:
            raise ValueError("Unexpected event structure in melody score.")
        new_melody.append((onset, melody_inst, kwargs.copy()))


    groups = defaultdict(list)
    for event in harmony:
        if len(event) == 3:
            onset, _, kwargs = event
        elif len(event) == 2:
            onset, kwargs = event
        else:
            raise ValueError("Unexpected event structure in harmony score.")
        groups[onset].append((onset, kwargs))

    new_harmony = []

    # for each chord sort by pitch and add delays.
    for onset, events in groups.items():

        # sort events by the 'pitch' value from kwargs (lowest to highest)
        sorted_events = sorted(events, key=lambda ev: ev[1]['pitch'])

        for idx, (orig_onset, kwargs) in enumerate(sorted_events):
            # add delay of rolled beats per note in the chord
            new_onset = orig_onset + rolled * idx
            new_harmony.append((new_onset, harmony_inst, kwargs.copy()))

    # combine melody and modified harmony events.
    combined = new_melody + new_harmony
    # sort the combined score by onset time.
    combined_sorted = sorted(combined, key=lambda ev: ev[0])
    
    return combined_sorted




def part4_rand(rhythm: np.ndarray,
                chords: List[Tuple[str, str]],
                chord_map: Dict[str, List[List[float]]],
                inst: Instrument,
                metronome: BasicMetronome,
                duration: float,
                random_state: np.random.RandomState) -> PlayableScore:
    """Generates an arpeggiated melody, with patterns randomly selected per
    chord according to random_state. Rhythm and duration are both in beats; when
    instantiating playable sound events with the provided instrument and
    duration, take care to convert this duration to seconds.

    Take care to only invoke random_state.rand() precisely once per chord.
    
    Returns: The constructed score.
    """

    # Ensure that each rhythm value is an exact multiple of 'duration'
    assert all([x % duration == 0 for x in rhythm]), "Duration must evenly divide rhythms"
    assert len(rhythm) == len(chords), "Chords and rhythm must have same length"
    
    # Compute seconds per beat directly from BPM.
    seconds_per_beat = 60.0 / metronome.bpm
    
    score = []
    current_beat = 0.0  # cumulative beat count

    # Loop over each chord event with its duration (in beats)
    for chord_event, chord_duration in zip(chords, rhythm):
        chord_root, chord_type = chord_event
        # Select a random pattern for the chord (exactly one random call per chord)
        pattern_list = chord_map[chord_type]
        pattern_index = int(random_state.rand() * len(pattern_list))
        chosen_pattern = pattern_list[pattern_index]

        # Determine the number of notes that fit in this chord event
        num_notes = int(chord_duration / duration)
        # Convert the chord root (string) to a pitch number
        root_pitch = pqh.pitch_name_to_pitch(chord_root)

        # For each note in the chord event:
        for j in range(num_notes):
            note_pitch = root_pitch + chosen_pattern[j % len(chosen_pattern)]
            # Compute the onset in beats for this note
            note_onset_beat = current_beat + j * duration
            # Convert onset and duration to seconds using seconds_per_beat
            note_onset_sec = note_onset_beat * seconds_per_beat
            note_duration_sec = duration * seconds_per_beat

            # Create the event with onset, instrument, and keyword arguments
            event = (note_onset_sec, inst, {"duration": note_duration_sec, "pitch": note_pitch})
            score.append(event)
        
        # Update the current beat by the chord's duration
        current_beat += chord_duration

    return score



def part5_composition() -> pq.Audio:
    """Create a composition using one or more songs from your choice in
    TheoryTab. TheoryTab data and FreeSound audio should be loaded from the
    "data/" directory, rather than using network requests.

    The above part4_rand and part2_harmonic_inst must be used at least once.

    Please limit file types to WAV or mp3.

    Returns: The generated final audio.
    """

    final_duration = 40.0  

    # load souces
    piano_loop = pq.Audio.from_file("data/piano_loop.mp3")
    hello_sample = pq.Audio.from_file("data/hello.wav")
    cash_register_sample = pq.Audio.from_file("data/cash_register.mp3")
    
    # piano loop sample rate for consistency
    sr = piano_loop.sample_rate
    final_samples = int(final_duration * sr)

# LAYER 1 : PIANO  
    piano_array = np.array(piano_loop)
    n_piano_samples = piano_array.shape[0]
    repeats = int(np.ceil(final_samples / n_piano_samples))

    piano_long = np.tile(piano_array, (repeats, 1))[:final_samples]
    

    # INIT FINAL MIX
    final_mix = np.zeros_like(piano_long, dtype=float)
    final_mix += 1.0 * piano_long

    # HELPER: function for layering sounds
    def add_sample(base_array, sample_array, onset_sec, weight=1.0):
        start_idx = int(onset_sec * sr)
        end_idx = start_idx + sample_array.shape[0]


        if end_idx > base_array.shape[0]:
            sample_array = sample_array[:base_array.shape[0] - start_idx]
            end_idx = base_array.shape[0]
        base_array[start_idx:end_idx] += weight * sample_array



# LAYER 2: Cash Register

    # HELPER: function for cash register instrument
    def cash_register_original_instrument(duration: float, **kwargs) -> pq.Audio:
        data = np.array(cash_register_sample)
        n_samples = int(np.floor(duration * sr))
        if data.shape[0] > n_samples:
            data = data[:n_samples]
        else:
            pad_width = n_samples - data.shape[0]
            if data.ndim == 1:
                data = np.pad(data, (0, pad_width))
            else:
                data = np.pad(data, ((0, pad_width), (0, 0)))
        return pq.Audio.from_array(data, sample_rate=sr)
    
    # cash register at onsets
    cash_hit_duration = 1.5  # clip 1.5s of audio

    cash_hit = cash_register_original_instrument(cash_hit_duration)
    cash_hit_array = np.array(cash_hit)

    for onset in [5, 15, 25, 35]:
        add_sample(final_mix, cash_hit_array, onset, weight=0.9)
    

# LAYER 3: Arpeggiated Section (part2_harmonic_inst req)

    # basic rising falling envelope
    simple_env = Envelope(np.array([0, 0.3, 1.0]), np.array([0, 1, 0]))
    arpeggio_h_envs = [simple_env for _ in range(4)]
    
    # Use part4_rand to generate an arpeggiated score
    chords = [("c4", "maj"), ("a3", "min"), ("f3", "maj"), ("g3", "maj")]

    # each chord is 4 beats long
    rhythm = np.array([4, 4, 4, 4]) 

    # each note is 0.5 beats long
    note_duration_beats = 0.5    
    chord_map = {
        "maj": [[0, 4, 7, 11], [0, 3, 7]],
        "min": [[0, 3, 7, 10], [0, 4, 7]]
    }

    # metronome for arpeggio: 120 bpm
    arp_metronome = BasicMetronome(120)

    # part2_harmonic_inst on envelope
    arpeggio_instrument = lambda duration, pitch, **kwargs: \
        part2_harmonic_inst(duration, pitch, harmonic_envs=arpeggio_h_envs, sample_rate=sr)
    
    arpeggio_score = part4_rand(
        rhythm=rhythm,
        chords=chords,
        chord_map=chord_map,
        inst=arpeggio_instrument,
        metronome=arp_metronome,
        duration=note_duration_beats,
        random_state=np.random.RandomState(42)
    )

    arpeggio_audio = render_score(arpeggio_score, arp_metronome)
    arpeggio_array = np.array(arpeggio_audio)

    # shift layer: Start @ 10s
    offset_samples = int(10 * sr)
    arpeggio_full = np.zeros((final_samples, arpeggio_array.shape[1]))
    end_idx = offset_samples + arpeggio_array.shape[0]
    if end_idx > final_samples:
        arpeggio_array = arpeggio_array[:final_samples - offset_samples]
        end_idx = final_samples
    arpeggio_full[offset_samples:end_idx] = arpeggio_array
    
    # arpeggio layer (NOTE: MAKE QUIERTER)
    final_mix += 0.1 * arpeggio_full


# LAYER 4: HELLO WORLD

    hello_array = np.array(hello_sample)
    hello_onsets = [0, 10, 20, 30, 38]

    #vary weights for each one
    hello_weights = [10.0, 3.0, 8.0, 3.0, 10.0]
    for onset, weight in zip(hello_onsets, hello_weights):
        add_sample(final_mix, hello_array, onset_sec=onset, weight=weight)
    


# LAYER 5: HELLO BACKGROUND LOOP

    # crop hello piece
    start_crop = int(0.3 * sr)
    end_crop = int(1.21 * sr)
    cropped_hello = hello_sample[start_crop:end_crop]
    cropped_hello_array = np.array(cropped_hello)

    # HELPER: function to speed up audio
    def speedup_audio(audio_array, speed_factor):
        orig_length = audio_array.shape[0]
        new_length = int(orig_length / speed_factor)

        # Create new indices at the desired shorter length.
        original_indices = np.linspace(0, orig_length - 1, num=orig_length)
        new_indices = np.linspace(0, orig_length - 1, num=new_length)
        if audio_array.ndim == 1:
            return np.interp(new_indices, original_indices, audio_array)
        else:
            channels = []
            for ch in range(audio_array.shape[1]):
                channels.append(np.interp(new_indices, original_indices, audio_array[:, ch]))
            return np.stack(channels, axis=1)

    # speed factor
    speed_factor = 2.0
    sped_up_hello_array = speedup_audio(cropped_hello_array, speed_factor)

    # loop sped up snipper
    n_sped_up = sped_up_hello_array.shape[0]
    background_hello_loop = np.tile(sped_up_hello_array, (int(np.ceil(final_samples / n_sped_up)), 1))[:final_samples]

    # add to mix
    final_mix += 1.0 * background_hello_loop



    # FINAL MIX
    max_val = np.max(np.abs(final_mix))
    if max_val > 1.0:
        final_mix = final_mix / max_val

    final_audio = pq.Audio.from_array(final_mix, sample_rate=sr)
    return final_audio


def permission_quiz() -> dict[str, bool]:
    import json
    import pathlib

    permission_file = pathlib.Path(__file__).parent / "permission.json"
    if not permission_file.exists():
        result = {
            "audio": False,
            "name": False,
            "notes": False,
            "code": False,
        }
        overall = input("Do we have permission to release your audio? (y/n): ")
        if overall.lower().strip() == "y":
            result["audio"] = True
            prompts = {
                "name": "Would you like to be credited by name? (y/n): ",
                "notes": "Can we share your intention as described in `answers.txt`? (y/n): ",
                "code": "Can we share your code? (y/n): ",
            }
            for key, prompt in prompts.items():
                result[key] = input(prompt).lower().strip() == "y"
        with open(permission_file, "w") as f:
            json.dump(result, f, indent=2)
    with open(permission_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    from pyquist.cli import play
    # Comment and uncomment the play (and/or write commands) as helpful for testing when completing this project


    ### TASK 1: Basic Instrument
    audio_unit = part1_inst(0.5, pqh.pitch_name_to_pitch("c4"))
    play(audio_unit)
    audio_unit.write("part1_inst.wav")


    ### TASK 1: Score with Basic Instrument
    onset_beats = [0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0]
    durations = [0.2] * 5 + [0.1] * 5
    pitches = ["c4", "d4", "eb4", "f4", "g4", "c4", "d4", "eb4", "f4", "g4"]
    pitches = [pqh.pitch_name_to_pitch(p) for p in pitches]
    metronome = BasicMetronome(120)
    score_handcrafted = part1_score(onset_beats, durations, pitches, part1_inst)
    audio_handcrafted = render_score(score_handcrafted, metronome)
    play(audio_handcrafted)
    audio_handcrafted.write("part1_score.wav")
    

    ### TASK 2: Envelope-shaped Instrument
    adsr = Envelope(np.array([0, 0.2, 0.3, 0.5, 1]), np.array([0, 1, 0.6, 0.6, 0]))
    audio_adsr = part2_inst(1, pqh.pitch_name_to_pitch("c4"), adsr)
    play(audio_adsr)
    audio_adsr.write("part2_adsr.wav")


    ### TASK 2: Score with Envelope-shaped Instrument
    handcrafted_env = Envelope(np.array([0, 0.01, 0.02, 0.08, 0.1]), np.array([0, 1, 0.8, 0.8, 0]))
    score_handcrafted_env = part2_score(onset_beats, durations, pitches, handcrafted_env, part2_inst)
    audio_handcrafted_env = render_score(score_handcrafted_env, metronome)
    play(audio_handcrafted_env)
    audio_handcrafted_env.write("part2_score_env.wav")

    
    ### TASK 2: Harmonic Instrument
    # pluck
    pluck_audio, pluck_meta = fetch("https://freesound.org/people/Skamos66/sounds/399468/")
    trumpet_audio, trumpet_meta = fetch("https://freesound.org/people/slothrop/sounds/48224/")
    recorder_audio, recorder_meta = fetch("https://freesound.org/people/cdonahueucsd/sounds/620964/")

    # speed up x2 for desired envelope length
    pluck_audio = pluck_audio[::2]
    trumpet_audio = trumpet_audio[::2]
    slow_durations = np.array(durations) * 4
    slow_metronome = BasicMetronome(90)
    # or trumpet_audio, pluck_audio, etc.
    ref_envelopes_recorder = get_harmonic_envelopes(recorder_audio)
    ref_envelopes_trumpet = get_harmonic_envelopes(trumpet_audio)
    ref_envelopes_pluck = get_harmonic_envelopes(pluck_audio)
    audio_harmonic = part2_harmonic_inst(1, pqh.pitch_name_to_pitch("c4"), ref_envelopes_recorder)
    play(audio_harmonic)
    audio_harmonic.write("part2_harmonic.wav")


    ### TASK 2: Score with Harmonic Instrument
    harmonic_score = part2_harmonic_score(onset_beats, slow_durations, pitches, recorder_audio)
    audio_harmonic_score = render_score(harmonic_score, slow_metronome)
    play(audio_harmonic_score)
    audio_harmonic_score.write("part2_harmonic_score.wav")

    
    ### TASK 3: Song Cover
    # AntiHero - Taylor Swift
    song_data = fetch_theorytab_json("https://hookpad.hooktheory.com/?idOfSong=ZOxVjA-Nxdq")
    song_info = theorytab_json_to_score(song_data)
    melody_env = Envelope(np.array([0, 0.85, 1]) * 0.15, np.array([1, 1, 0]))
    # can swap around instruments here, as long as the instrument takes pitch and duration as keyword args
    def melody_harmonic_inst(*args, **kwargs): return 0.3 * part2_harmonic_inst(*args, **kwargs, harmonic_envs=ref_envelopes_recorder, overall_env=melody_env)
    def harmony_harmonic_inst(*args, **kwargs): return 0.12 * part2_harmonic_inst(*args, **kwargs, harmonic_envs=ref_envelopes_pluck)
    score_cover = part3_cover(song_info[1], song_info[2], melody_harmonic_inst, harmony_harmonic_inst, rolled=0.25)
    audio_cover = render_score(score_cover, song_info[0])
    play(audio_cover)
    audio_cover.write("part3_cover.wav")


    ### TASK 4: Random Generation
    maj_patterns = [[0, 4, 7, 11, 7, 4], [0, 4, 7, 11, 12], [12, 11, 7, 4]]
    min_patterns = [[0, 3, 7, 10, 7, 3], [0, 3, 7, 10, 12], [12, 10, 7, 3]]
    chord_map = {"maj": maj_patterns, "min": min_patterns}
    chords = [("f4", "maj"), ("f4", "maj"), 
              ("e4", "min"), 
              ("e4", "min"), ("eb4", "min"),
              ("d4", "min"), ("d4", "min"),
              ("c4", "maj"), ("c4", "maj"), ("c4", "maj"), ("c4", "maj")]
    rhythm = np.array([4, 4, 4, 2, 2, 4, 4, 4, 1.5, 1.5, 1])

    rand_metronome = BasicMetronome(180)
    score_rand = part4_rand(rhythm=rhythm,
                            chords=chords,
                            chord_map=chord_map,
                            inst=melody_harmonic_inst,
                            metronome=rand_metronome,
                            duration=0.5,
                            random_state=np.random.RandomState(1))
    audio_rand = render_score(score_rand, rand_metronome)
    play(audio_rand)
    audio_rand.write("part4_rand.wav")


    ### TASK 5: Composition
    audio_composition = part5_composition()
    play(audio_composition)
    audio_composition.write("part5_composition.wav")

    # Run this once before submitting the assignment; comment out when complete
    permission_quiz()
