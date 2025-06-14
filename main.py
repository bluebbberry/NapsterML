import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pretty_midi
import glob
import os
import pickle
import random
from collections import defaultdict
import matplotlib.pyplot as plt


class MIDISongGenerator:
    def __init__(self, sequence_length=64, fs=16):
        self.sequence_length = sequence_length
        self.fs = fs  # Sampling frequency for note representation
        self.note_range = 128  # MIDI note range (0-127)
        self.velocity_range = 128  # MIDI velocity range (0-127)
        self.model = None
        self.training_data = []

    def load_midi_files(self, midi_folder):
        """Load and process MIDI files from a folder"""
        print(f"Loading MIDI files from: {midi_folder}")
        midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))

        songs = []
        failed_files = []

        for midi_file in midi_files:
            try:
                song_data = self.midi_to_piano_roll(midi_file)
                if song_data is not None and len(song_data) > self.sequence_length:
                    songs.append(song_data)
                    print(f"✓ Loaded: {os.path.basename(midi_file)} - {len(song_data)} time steps")
            except Exception as e:
                failed_files.append(midi_file)
                print(f"✗ Failed: {os.path.basename(midi_file)} - {str(e)}")

        print(f"\nSuccessfully loaded: {len(songs)} songs")
        print(f"Failed to load: {len(failed_files)} songs")

        return songs

    def midi_to_piano_roll(self, midi_file):
        """Convert MIDI file to piano roll representation"""
        try:
            pm = pretty_midi.PrettyMIDI(midi_file)

            # Filter out drum tracks and get melodic instruments
            melodic_instruments = [inst for inst in pm.instruments if not inst.is_drum]

            if not melodic_instruments:
                return None

            # Get song duration and create time grid
            end_time = pm.get_end_time()
            if end_time == 0:
                return None

            # Create piano roll with time steps
            time_step = 1.0 / self.fs  # Duration of each time step
            n_time_steps = int(np.ceil(end_time / time_step))

            # Piano roll: [time_steps, note_features]
            # Features: [pitch, velocity, instrument_id]
            piano_roll = []

            for t in range(n_time_steps):
                current_time = t * time_step
                time_slice = []

                for inst_idx, instrument in enumerate(melodic_instruments):
                    for note in instrument.notes:
                        if note.start <= current_time < note.end:
                            # Note is active at this time
                            time_slice.append({
                                'pitch': note.pitch,
                                'velocity': note.velocity,
                                'instrument': inst_idx,
                                'duration': note.end - note.start
                            })

                # Convert to numerical representation
                if time_slice:
                    # Take the highest priority notes (highest velocity)
                    time_slice.sort(key=lambda x: x['velocity'], reverse=True)
                    # Limit to top 4 simultaneous notes to keep complexity manageable
                    time_slice = time_slice[:4]

                    # Encode as vector
                    encoded_slice = []
                    for note in time_slice:
                        encoded_slice.extend([
                            note['pitch'] / 127.0,  # Normalized pitch
                            note['velocity'] / 127.0,  # Normalized velocity
                            note['instrument'] / max(1, len(melodic_instruments) - 1),  # Normalized instrument
                            min(note['duration'], 4.0) / 4.0  # Normalized duration (capped at 4 seconds)
                        ])

                    # Pad to fixed size (4 notes * 4 features = 16 features)
                    while len(encoded_slice) < 16:
                        encoded_slice.append(0.0)

                    piano_roll.append(encoded_slice[:16])
                else:
                    # Silent time step
                    piano_roll.append([0.0] * 16)

            return np.array(piano_roll)

        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            return None

    def prepare_training_data(self, songs):
        """Prepare training sequences from songs"""
        print("Preparing training data...")

        X, y = [], []

        for song in songs:
            # Create overlapping sequences
            for i in range(0, len(song) - self.sequence_length, self.sequence_length // 4):
                if i + self.sequence_length < len(song):
                    input_sequence = song[i:i + self.sequence_length]
                    target_sequence = song[i + 1:i + self.sequence_length + 1]

                    X.append(input_sequence)
                    y.append(target_sequence)

        X = np.array(X)
        y = np.array(y)

        print(f"Created {len(X)} training sequences")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")

        return X, y

    def build_model(self, lstm_units=256, dropout_rate=0.3):
        """Build the LSTM model for music generation"""
        model = Sequential([
            # Input layer
            LSTM(lstm_units, return_sequences=True, input_shape=(self.sequence_length, 16)),
            Dropout(dropout_rate),

            # Hidden layers
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout_rate),

            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),

            # Output layer
            Dense(32, activation='relu'),
            Dropout(dropout_rate),
            Dense(16, activation='sigmoid')  # Output same dimension as input
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        """Train the music generation model"""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Callbacks
        checkpoint = ModelCheckpoint(
            'best_music_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )

        # Train the model
        history = self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[checkpoint, early_stop, reduce_lr],
            verbose=1
        )

        return history

    def generate_music(self, seed_sequence=None, length=256, temperature=1.0):
        """Generate new music sequence"""
        if self.model is None:
            raise ValueError("Model not trained. Train the model first.")

        if seed_sequence is None:
            # Use random seed from training data
            if self.training_data:
                seed_sequence = random.choice(self.training_data)[:self.sequence_length]
            else:
                seed_sequence = np.random.random((self.sequence_length, 16))

        generated = seed_sequence.copy()

        for _ in range(length):
            # Prepare input
            input_seq = generated[-self.sequence_length:].reshape(1, self.sequence_length, 16)

            # Predict next time step
            prediction = self.model.predict(input_seq, verbose=0)[0]

            # Apply temperature for creativity
            if temperature != 1.0:
                prediction = np.power(prediction, 1.0 / temperature)

            # Add some randomness
            noise = np.random.normal(0, 0.1, prediction.shape)
            prediction = np.clip(prediction + noise * temperature, 0, 1)

            generated = np.vstack([generated, prediction[-1].reshape(1, 16)])

        return generated[self.sequence_length:]

    def piano_roll_to_midi(self, piano_roll, filename='generated_song.mid', tempo=120):
        """Convert piano roll back to MIDI file"""
        # Create MIDI file
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Create instruments
        instruments = []
        for i in range(4):  # Support up to 4 instruments
            instrument = pretty_midi.Instrument(program=i)
            instruments.append(instrument)

        time_step = 1.0 / self.fs

        # Process each time step
        for t, time_slice in enumerate(piano_roll):
            current_time = t * time_step

            # Decode the time slice (4 notes * 4 features each)
            for note_idx in range(4):
                start_idx = note_idx * 4

                pitch = time_slice[start_idx] * 127
                velocity = time_slice[start_idx + 1] * 127
                instrument_id = int(time_slice[start_idx + 2] * 3)  # 0-3
                duration = time_slice[start_idx + 3] * 4  # Max 4 seconds

                # Only create note if it has significant values
                if pitch > 0.1 and velocity > 0.1 and duration > 0.1:
                    note = pretty_midi.Note(
                        velocity=max(1, int(velocity)),
                        pitch=max(0, min(127, int(pitch))),
                        start=current_time,
                        end=current_time + duration
                    )

                    # Add to appropriate instrument
                    if instrument_id < len(instruments):
                        instruments[instrument_id].notes.append(note)

        # Add instruments to MIDI
        for instrument in instruments:
            if instrument.notes:  # Only add if it has notes
                pm.instruments.append(instrument)

        # Save MIDI file
        pm.write(filename)
        print(f"Generated MIDI saved as: {filename}")
        return filename

    def create_song_with_structure(self, sections=None, filename='structured_song.mid'):
        """Generate a song with traditional structure"""
        if sections is None:
            sections = {
                'intro': 32,
                'verse1': 64,
                'chorus1': 48,
                'verse2': 64,
                'chorus2': 48,
                'bridge': 32,
                'chorus3': 48,
                'outro': 24
            }

        full_song = None

        for section_name, section_length in sections.items():
            print(f"Generating {section_name} ({section_length} steps)...")

            # Vary temperature by section
            temp_map = {
                'intro': 0.8, 'verse1': 0.7, 'chorus1': 0.9,
                'verse2': 0.7, 'chorus2': 0.9, 'bridge': 1.2,
                'chorus3': 0.9, 'outro': 0.6
            }

            temperature = temp_map.get(section_name, 0.8)
            section_music = self.generate_music(length=section_length, temperature=temperature)

            if full_song is None:
                full_song = section_music
            else:
                full_song = np.vstack([full_song, section_music])

        # Convert to MIDI
        midi_file = self.piano_roll_to_midi(full_song, filename)
        return midi_file, full_song

    def analyze_training_data(self, songs):
        """Analyze the loaded songs for insights"""
        print("\nAnalyzing training data...")

        total_duration = sum(len(song) for song in songs) / self.fs
        avg_song_length = np.mean([len(song) for song in songs]) / self.fs

        print(f"Total music duration: {total_duration:.1f} seconds ({total_duration / 60:.1f} minutes)")
        print(f"Average song length: {avg_song_length:.1f} seconds")
        print(f"Number of songs: {len(songs)}")

        # Analyze note distributions
        all_pitches = []
        all_velocities = []

        for song in songs:
            for time_step in song:
                for note_idx in range(4):
                    start_idx = note_idx * 4
                    pitch = time_step[start_idx] * 127
                    velocity = time_step[start_idx + 1] * 127

                    if pitch > 0.1 and velocity > 0.1:
                        all_pitches.append(pitch)
                        all_velocities.append(velocity)

        if all_pitches:
            print(f"Pitch range: {min(all_pitches):.0f} - {max(all_pitches):.0f}")
            print(f"Average pitch: {np.mean(all_pitches):.0f}")
            print(f"Average velocity: {np.mean(all_velocities):.0f}")

    def save_model(self, filepath='midi_generator.pkl'):
        """Save the trained model and metadata"""
        model_data = {
            'sequence_length': self.sequence_length,
            'fs': self.fs,
            'note_range': self.note_range,
            'velocity_range': self.velocity_range
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        if self.model:
            self.model.save('midi_model.h5')

        print(f"Model saved to {filepath}")

    def load_model(self, filepath='midi_generator.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.sequence_length = model_data['sequence_length']
        self.fs = model_data['fs']
        self.note_range = model_data['note_range']
        self.velocity_range = model_data['velocity_range']

        self.model = tf.keras.models.load_model('midi_model.h5')
        print("Model loaded successfully")


def main():
    """Main training pipeline"""
    print("MIDI Song Generator - Training on Real Songs")
    print("=" * 50)

    # Initialize generator
    generator = MIDISongGenerator(sequence_length=64, fs=16)

    # Load MIDI files
    midi_folder = input("Enter path to MIDI files folder (or press Enter for default 'midi_files'): ").strip()
    if not midi_folder:
        midi_folder = 'midi_files'

    if not os.path.exists(midi_folder):
        print(f"Creating example folder: {midi_folder}")
        os.makedirs(midi_folder)
        print(f"Please add MIDI files to '{midi_folder}' folder and run again.")
        print("You can download free MIDI files from:")
        print("- https://www.midiworld.com/")
        print("- https://freemidi.org/")
        print("- https://www.piano-midi.de/")
        return

    # Load songs
    songs = generator.load_midi_files(midi_folder)

    if not songs:
        print("No MIDI files found or loaded successfully!")
        return

    # Analyze the data
    generator.analyze_training_data(songs)
    generator.training_data = songs

    # Prepare training data
    X, y = generator.prepare_training_data(songs)

    # Build model
    print("\nBuilding model...")
    model = generator.build_model(lstm_units=512, dropout_rate=0.3)
    print(model.summary())

    # Train model
    print("\nTraining model...")
    history = generator.train(X, y, epochs=200, batch_size=16)

    # Save model
    generator.save_model()

    # Generate new songs
    print("\nGenerating new songs...")

    # Generate multiple songs with different styles
    temperatures = [0.7, 0.9, 1.1]

    for i, temp in enumerate(temperatures):
        print(f"\nGenerating song {i + 1} (temperature={temp})...")

        # Generate simple song
        generated_music = generator.generate_music(length=200, temperature=temp)
        filename = f'generated_song_{i + 1}_temp_{temp}.mid'
        generator.piano_roll_to_midi(generated_music, filename)

        # Generate structured song
        structured_filename = f'structured_song_{i + 1}_temp_{temp}.mid'
        generator.create_song_with_structure(filename=structured_filename)

    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("Generated MIDI files are ready to play!")
    print("\nFiles created:")
    print("- Multiple generated songs with different creativity levels")
    print("- Structured songs with intro/verse/chorus/bridge/outro")
    print("\nYou can:")
    print("- Play them in any MIDI player")
    print("- Import into DAW software (FL Studio, Ableton, etc.)")
    print("- Convert to audio files")


if __name__ == "__main__":
    main()

# Example usage after training:
"""
# Load trained model and generate songs
generator = MIDISongGenerator()
generator.load_model('midi_generator.pkl')

# Generate songs in different styles
classical_style = generator.generate_music(length=300, temperature=0.6)
jazz_style = generator.generate_music(length=250, temperature=1.2)
pop_style = generator.generate_music(length=200, temperature=0.8)

# Save as MIDI files
generator.piano_roll_to_midi(classical_style, 'classical_generated.mid')
generator.piano_roll_to_midi(jazz_style, 'jazz_generated.mid')
generator.piano_roll_to_midi(pop_style, 'pop_generated.mid')

# Create full albums
for i in range(10):
    filename = f'album_track_{i+1}.mid'
    generator.create_song_with_structure(filename=filename)
    print(f"Generated: {filename}")
"""
