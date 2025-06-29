# P2P Music Generation Network MVP
# A distributed system for collaborative music model training and generation

import asyncio
import json
import hashlib
import time
import random
import os
import pickle
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import socket
import threading
from collections import defaultdict
import numpy as np


# Mock implementations for external dependencies
class MockFlowerClient:
    """Mock Flower FL client with basic training logic"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.model_weights = np.random.random((100, 50))  # Mock model weights
        self.learning_rate = 0.01

    def get_parameters(self):
        return self.model_weights.flatten()

    def set_parameters(self, parameters):
        self.model_weights = np.array(parameters).reshape((100, 50))

    def fit(self, training_data):
        # Basic training simulation: adjust weights based on music features
        if not training_data:
            return 0, {"loss": 1.0}

        # Extract features from training data and do simple gradient descent
        total_loss = 0
        for data in training_data:
            features = data.get('features', {})
            # Convert music features to numerical values
            tempo = features.get('tempo', 120) / 200.0  # Normalize tempo
            energy = features.get('energy', 0.5)
            key_map = {'C': 0, 'D': 0.14, 'E': 0.28, 'F': 0.42, 'G': 0.57, 'A': 0.71, 'B': 0.85}
            key_val = key_map.get(features.get('key', 'C'), 0)

            # Create target vector from features
            target = np.array([tempo, energy, key_val] * 17)[:50]  # Pad to 50 features

            # Simple loss calculation and weight update
            for i in range(self.model_weights.shape[0]):
                prediction = np.dot(self.model_weights[i], target)
                loss = (prediction - tempo) ** 2  # Simple MSE loss
                total_loss += loss

                # Gradient descent update
                gradient = 2 * (prediction - tempo) * target
                self.model_weights[i] -= self.learning_rate * gradient

        avg_loss = total_loss / (len(training_data) * self.model_weights.shape[0])
        return len(training_data), {"loss": float(avg_loss)}

    def evaluate(self, test_data):
        return len(test_data), {"accuracy": random.uniform(0.7, 0.9)}


class MockMusicGenerator:
    """Mock music generation model that learns from training"""

    def __init__(self, weights=None):
        self.weights = weights or np.random.random((100, 50))
        self.learned_genres = []
        self.learned_patterns = {}

    def update_from_training(self, training_results: List[Dict]):
        """Update generator based on aggregated training results"""
        # Extract genre patterns from training
        genre_counts = defaultdict(int)
        tempo_ranges = defaultdict(list)

        for result in training_results:
            if 'genres' in result:
                for genre in result['genres']:
                    genre_counts[genre] += 1
            if 'tempo_patterns' in result:
                for genre, tempos in result['tempo_patterns'].items():
                    tempo_ranges[genre].extend(tempos)

        self.learned_genres = list(genre_counts.keys())
        self.learned_patterns = {
            'genres': dict(genre_counts),
            'tempo_ranges': {k: (min(v), max(v)) if v else (120, 140)
                             for k, v in tempo_ranges.items()}
        }

    def generate_song(self, style_params: Dict) -> Dict:
        """Generate a song using learned patterns"""
        # Use learned patterns if available
        if self.learned_genres and random.random() > 0.3:
            genre = random.choice(self.learned_genres)
        else:
            genre = style_params.get('genre', random.choice(["pop", "rock", "jazz", "electronic"]))

        # Use learned tempo patterns
        if genre in self.learned_patterns.get('tempo_ranges', {}):
            tempo_min, tempo_max = self.learned_patterns['tempo_ranges'][genre]
            tempo = random.randint(int(tempo_min), int(tempo_max))
        else:
            tempo = style_params.get('tempo', random.randint(60, 180))

        # Generate song with learned characteristics
        song_features = np.mean(self.weights, axis=0)
        complexity = np.std(song_features)

        return {
            "title": f"AI Generated - {genre.title()} #{random.randint(1000, 9999)}",
            "duration": random.randint(120, 300),
            "genre": genre,
            "features": {
                "tempo": tempo,
                "key": random.choice(["C", "D", "E", "F", "G", "A", "B"]),
                "mood": random.choice(["happy", "sad", "energetic", "calm"]),
                "complexity": float(complexity),
                "energy": min(1.0, float(np.max(song_features)))
            },
            "audio_data": f"generated_audio_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}",
            "model_signature": hashlib.md5(self.weights.tobytes()).hexdigest()[:16]
        }


class NodeType(Enum):
    PEER = "peer"
    COORDINATOR = "coordinator"


@dataclass
class Song:
    title: str
    artist: str
    duration: int
    genre: str
    audio_features: Dict
    file_hash: str

    def to_training_data(self):
        """Convert song to training data format"""
        return {
            "features": self.audio_features,
            "metadata": {
                "genre": self.genre,
                "duration": self.duration
            }
        }


@dataclass
class Node:
    node_id: str
    ip_address: str
    port: int
    node_type: NodeType
    last_seen: float

    def to_dict(self):
        return asdict(self)


@dataclass
class TrainingSession:
    session_id: str
    coordinator_id: str
    participants: Set[str]
    status: str
    created_at: float
    model_version: int


class LocalStorage:
    """Handles local file storage for generated songs and models"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.base_dir = f"p2p_music_node_{node_id}"
        self.songs_dir = os.path.join(self.base_dir, "generated_songs")
        self.models_dir = os.path.join(self.base_dir, "models")
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.songs_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def save_song(self, song: Dict) -> str:
        """Save a generated song locally"""
        filename = f"{song['title'].replace(' ', '_')}_{int(time.time())}.json"
        filepath = os.path.join(self.songs_dir, filename)

        # Add local metadata
        song_data = song.copy()
        song_data['saved_at'] = time.time()
        song_data['local_path'] = filepath

        with open(filepath, 'w') as f:
            json.dump(song_data, f, indent=2)

        print(f"Saved song: {song['title']} -> {filepath}")
        return filepath

    def save_model(self, model_weights: np.ndarray, version: int) -> str:
        """Save model weights locally"""
        filename = f"model_v{version}_{int(time.time())}.pkl"
        filepath = os.path.join(self.models_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(model_weights, f)

        print(f"Saved model version {version} -> {filepath}")
        return filepath

    def load_saved_songs(self) -> List[Dict]:
        """Load all locally saved songs"""
        songs = []
        if os.path.exists(self.songs_dir):
            for filename in os.listdir(self.songs_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.songs_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            song = json.load(f)
                            songs.append(song)
                    except Exception as e:
                        print(f"Error loading song {filename}: {e}")
        return songs

    def get_storage_stats(self) -> Dict:
        """Get local storage statistics"""
        song_count = len([f for f in os.listdir(self.songs_dir) if f.endswith('.json')]) if os.path.exists(
            self.songs_dir) else 0
        model_count = len([f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]) if os.path.exists(
            self.models_dir) else 0

        return {
            'songs_stored': song_count,
            'models_stored': model_count,
            'storage_path': self.base_dir
        }


class P2PNetworkManager:
    """Handles P2P node discovery and communication"""

    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.port = port
        self.known_nodes: Dict[str, Node] = {}
        self.is_running = False
        self.discovery_socket = None

    async def start_discovery(self):
        """Start P2P node discovery using UDP broadcast"""
        self.is_running = True
        self.discovery_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.discovery_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.discovery_socket.bind(('', self.port))

        # Start discovery threads
        threading.Thread(target=self._discovery_listener, daemon=True).start()
        threading.Thread(target=self._discovery_broadcaster, daemon=True).start()

    def _discovery_listener(self):
        """Listen for discovery messages from other nodes"""
        while self.is_running:
            try:
                data, addr = self.discovery_socket.recvfrom(1024)
                message = json.loads(data.decode())

                if message['type'] == 'discovery' and message['node_id'] != self.node_id:
                    node = Node(
                        node_id=message['node_id'],
                        ip_address=addr[0],
                        port=message['port'],
                        node_type=NodeType(message['node_type']),
                        last_seen=time.time()
                    )
                    self.known_nodes[node.node_id] = node
                    print(f"Discovered node: {node.node_id} ({node.node_type.value})")

            except Exception as e:
                print(f"Error in discovery listener: {e}")

    def _discovery_broadcaster(self):
        """Broadcast discovery messages to find other nodes"""
        while self.is_running:
            try:
                message = {
                    'type': 'discovery',
                    'node_id': self.node_id,
                    'port': self.port,
                    'node_type': 'peer',
                    'timestamp': time.time()
                }

                self.discovery_socket.sendto(
                    json.dumps(message).encode(),
                    ('<broadcast>', self.port)
                )
                time.sleep(30)  # Broadcast every 30 seconds

            except Exception as e:
                print(f"Error in discovery broadcaster: {e}")

    def get_coordinators(self) -> List[Node]:
        """Get list of known coordinator nodes"""
        return [node for node in self.known_nodes.values()
                if node.node_type == NodeType.COORDINATOR]

    def get_peers(self) -> List[Node]:
        """Get list of known peer nodes"""
        return [node for node in self.known_nodes.values()
                if node.node_type == NodeType.PEER]


class MusicLibrary:
    """Manages local music collection and training data"""

    def __init__(self):
        self.songs: Dict[str, Song] = {}
        self.training_data: List[Dict] = []

    def add_song(self, song: Song):
        """Add a song to the library"""
        self.songs[song.file_hash] = song
        training_sample = song.to_training_data()
        self.training_data.append(training_sample)
        print(f"Added song: {song.title} by {song.artist}")

    def get_training_data(self) -> List[Dict]:
        """Get training data for federated learning"""
        return self.training_data

    def get_training_summary(self) -> Dict:
        """Get summary of training data for model learning"""
        if not self.training_data:
            return {}

        genres = [data['metadata']['genre'] for data in self.training_data]
        tempos = []
        for data in self.training_data:
            if 'tempo' in data['features']:
                tempos.append(data['features']['tempo'])

        tempo_by_genre = defaultdict(list)
        for data in self.training_data:
            genre = data['metadata']['genre']
            if 'tempo' in data['features']:
                tempo_by_genre[genre].append(data['features']['tempo'])

        return {
            'genres': list(set(genres)),
            'tempo_patterns': dict(tempo_by_genre),
            'total_songs': len(self.training_data)
        }

    def get_song_count(self) -> int:
        """Get total number of songs in library"""
        return len(self.songs)


class DistributedTrainingManager:
    """Manages distributed training using Flower AI"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.fl_client = MockFlowerClient(node_id)
        self.current_session: Optional[TrainingSession] = None

    async def join_training_session(self, session: TrainingSession, training_data: List[Dict]):
        """Join a federated learning training session"""
        print(f"Joining training session {session.session_id}")

        # Simulate federated learning round with actual training
        self.current_session = session

        # Fit model on local data (now with real training logic)
        loss, metrics = self.fl_client.fit(training_data)
        print(f"Local training completed. Data points: {loss}, Loss: {metrics.get('loss', 'N/A'):.4f}")

        # Get updated model parameters
        parameters = self.fl_client.get_parameters()

        return {
            'node_id': self.node_id,
            'parameters': parameters.tolist(),
            'data_size': len(training_data),
            'metrics': metrics
        }

    def get_model_weights(self):
        """Get current model weights"""
        return self.fl_client.get_parameters()


class CoordinatorNode:
    """Central coordination node for training orchestration and song generation"""

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.node_type = NodeType.COORDINATOR
        self.active_sessions: Dict[str, TrainingSession] = {}
        self.global_model = MockMusicGenerator()
        self.generated_songs: List[Dict] = []
        self.local_storage = LocalStorage(node_id)

    async def orchestrate_training(self, participants: List[str]) -> str:
        """Orchestrate a federated learning training session"""
        session_id = hashlib.md5(f"{time.time()}{self.node_id}".encode()).hexdigest()[:12]

        session = TrainingSession(
            session_id=session_id,
            coordinator_id=self.node_id,
            participants=set(participants),
            status="active",
            created_at=time.time(),
            model_version=len(self.active_sessions)
        )

        self.active_sessions[session_id] = session
        print(f"Started training session {session_id} with {len(participants)} participants")

        return session_id

    def aggregate_model_updates(self, session_id: str, updates: List[Dict]) -> Dict:
        """Aggregate model updates from participants using federated averaging"""
        if session_id not in self.active_sessions:
            return {"error": "Session not found"}

        # Simple federated averaging
        total_data_size = sum(update['data_size'] for update in updates)

        if total_data_size == 0:
            return {"error": "No training data"}

        # Weighted average of parameters
        aggregated_params = np.zeros_like(np.array(updates[0]['parameters']))

        for update in updates:
            weight = update['data_size'] / total_data_size
            aggregated_params += weight * np.array(update['parameters'])

        # Update global model with aggregated parameters
        self.global_model.weights = aggregated_params.reshape((100, 50))

        # Collect training summaries from updates for model learning
        training_summaries = []
        for update in updates:
            if 'training_summary' in update:
                training_summaries.append(update['training_summary'])

        # Update generator with learned patterns
        if training_summaries:
            self.global_model.update_from_training(training_summaries)

        # Mark session as completed
        self.active_sessions[session_id].status = "completed"

        print(f"Model aggregation completed for session {session_id}")
        avg_loss = np.mean([update['metrics'].get('loss', 0) for update in updates])
        print(f"Average training loss: {avg_loss:.4f}")

        return {
            "session_id": session_id,
            "model_version": self.active_sessions[session_id].model_version,
            "participants": len(updates),
            "avg_loss": float(avg_loss)
        }

    def generate_songs(self, count: int = 5, style_params: Dict = None) -> List[Dict]:
        """Generate songs using the trained global model"""
        if style_params is None:
            style_params = {}

        new_songs = []
        for _ in range(count):
            song = self.global_model.generate_song(style_params)
            song['generated_by'] = self.node_id
            song['model_version'] = len(self.active_sessions)
            song['timestamp'] = time.time()
            new_songs.append(song)

        self.generated_songs.extend(new_songs)
        print(f"Generated {count} new songs")
        return new_songs


class PeerNode:
    """Peer node that participates in training and downloads generated music"""

    def __init__(self, node_id: str, port: int):
        self.node_id = node_id
        self.node_type = NodeType.PEER
        self.network_manager = P2PNetworkManager(node_id, port)
        self.music_library = MusicLibrary()
        self.training_manager = DistributedTrainingManager(node_id)
        self.local_storage = LocalStorage(node_id)
        self.downloaded_songs: List[Dict] = []

    async def start(self):
        """Start the peer node"""
        print(f"Starting peer node {self.node_id}")
        await self.network_manager.start_discovery()

        # Add some mock songs to library
        await self._add_sample_songs()

        # Load previously saved songs
        saved_songs = self.local_storage.load_saved_songs()
        print(f"Loaded {len(saved_songs)} previously saved songs")

    async def _add_sample_songs(self):
        """Add sample songs for demonstration with realistic features"""
        sample_data = [
            {"title": "Upbeat Pop Track", "artist": "AI Artist A", "genre": "pop",
             "tempo": 128, "key": "C", "energy": 0.8},
            {"title": "Chill Electronic", "artist": "AI Artist B", "genre": "electronic",
             "tempo": 100, "key": "Am", "energy": 0.6},
            {"title": "Rock Anthem", "artist": "AI Artist C", "genre": "rock",
             "tempo": 140, "key": "E", "energy": 0.9},
        ]

        for i, data in enumerate(sample_data):
            song = Song(
                title=data["title"],
                artist=data["artist"],
                duration=random.randint(180, 240),
                genre=data["genre"],
                audio_features={
                    "tempo": data["tempo"],
                    "key": data["key"],
                    "energy": data["energy"]
                },
                file_hash=f"hash_{self.node_id}_{i}"
            )
            self.music_library.add_song(song)

    async def participate_in_training(self):
        """Find and participate in training sessions"""
        coordinators = self.network_manager.get_coordinators()

        if not coordinators:
            print("No coordinators found for training")
            return

        # Join training with first available coordinator
        coordinator = coordinators[0]
        print(f"Attempting to join training with coordinator {coordinator.node_id}")

        training_data = self.music_library.get_training_data()

        if training_data:
            # Create training session
            mock_session = TrainingSession(
                session_id="mock_session",
                coordinator_id=coordinator.node_id,
                participants={self.node_id},
                status="active",
                created_at=time.time(),
                model_version=1
            )

            result = await self.training_manager.join_training_session(mock_session, training_data)

            # Add training summary to result
            result['training_summary'] = self.music_library.get_training_summary()

            print(f"Training participation result: {result['data_size']} songs contributed")
            return result

    def download_generated_songs(self, songs: List[Dict]):
        """Download and save generated songs locally"""
        newly_downloaded = 0
        for song in songs:
            if song not in self.downloaded_songs:
                self.downloaded_songs.append(song)
                # Save song locally
                self.local_storage.save_song(song)
                newly_downloaded += 1
                print(f"Downloaded & Saved: {song['title']} ({song['genre']})")

        if newly_downloaded > 0:
            stats = self.local_storage.get_storage_stats()
            print(f"Total songs stored locally: {stats['songs_stored']}")

    def get_local_library_stats(self) -> Dict:
        """Get statistics about local music library and storage"""
        storage_stats = self.local_storage.get_storage_stats()
        library_stats = self.music_library.get_training_summary()

        return {
            **storage_stats,
            **library_stats,
            'downloaded_songs': len(self.downloaded_songs)
        }


class P2PMusicNetwork:
    """Main application class that orchestrates the entire system"""

    def __init__(self):
        self.nodes: Dict[str, PeerNode] = {}
        self.coordinators: Dict[str, CoordinatorNode] = {}

    def create_peer_node(self, node_id: str, port: int) -> PeerNode:
        """Create a new peer node"""
        node = PeerNode(node_id, port)
        self.nodes[node_id] = node
        return node

    def create_coordinator_node(self, node_id: str) -> CoordinatorNode:
        """Create a new coordinator node"""
        coordinator = CoordinatorNode(node_id)
        self.coordinators[node_id] = coordinator
        return coordinator

    async def simulate_network_activity(self):
        """Simulate network activity for demonstration"""
        print("=== P2P Music Generation Network Simulation ===\n")

        # Create nodes
        peer1 = self.create_peer_node("peer_001", 8001)
        peer2 = self.create_peer_node("peer_002", 8002)
        coordinator = self.create_coordinator_node("coord_001")

        # Start peer nodes
        await peer1.start()
        await peer2.start()

        # Simulate some delay for network discovery
        await asyncio.sleep(2)

        # Simulate training orchestration
        print("\n--- Starting Federated Training ---")
        session_id = await coordinator.orchestrate_training(["peer_001", "peer_002"])

        # Peers participate in training
        peer1_result = await peer1.participate_in_training()
        peer2_result = await peer2.participate_in_training()

        # Simulate model updates aggregation with training summaries
        mock_updates = [
            {
                'node_id': 'peer_001',
                'parameters': peer1.training_manager.get_model_weights().tolist(),
                'data_size': peer1.music_library.get_song_count(),
                'metrics': peer1_result['metrics'] if peer1_result else {'loss': 0.5},
                'training_summary': peer1.music_library.get_training_summary()
            },
            {
                'node_id': 'peer_002',
                'parameters': peer2.training_manager.get_model_weights().tolist(),
                'data_size': peer2.music_library.get_song_count(),
                'metrics': peer2_result['metrics'] if peer2_result else {'loss': 0.4},
                'training_summary': peer2.music_library.get_training_summary()
            }
        ]

        result = coordinator.aggregate_model_updates(session_id, mock_updates)
        print(f"Aggregation result: {result}")

        # Generate new songs
        print("\n--- Generating New Music ---")
        generated_songs = coordinator.generate_songs(3, {"genre": "electronic", "tempo": 128})

        # Distribute to peers
        print("\n--- Distributing Generated Songs ---")
        peer1.download_generated_songs(generated_songs)
        peer2.download_generated_songs(generated_songs)

        # Print summary
        print(f"\n=== Network Summary ===")
        print(f"Peers: {len(self.nodes)}")
        print(f"Coordinators: {len(self.coordinators)}")
        print(f"Songs generated: {len(generated_songs)}")

        # Show detailed peer statistics
        for node_id, peer in self.nodes.items():
            stats = peer.get_local_library_stats()
            print(f"\n{node_id} Stats:")
            print(f"  - Training songs: {stats.get('total_songs', 0)}")
            print(f"  - Downloaded songs: {stats['downloaded_songs']}")
            print(f"  - Songs saved locally: {stats['songs_stored']}")
            print(f"  - Storage path: {stats['storage_path']}")
            if stats.get('genres'):
                print(f"  - Music genres: {', '.join(stats['genres'])}")

        # Save final model state
        final_model_path = coordinator.local_storage.save_model(
            coordinator.global_model.weights,
            len(coordinator.active_sessions)
        )
        print(f"\nFinal trained model saved: {final_model_path}")

        print(f"\n🎵 Generated songs are saved in each peer's local directory!")
        print(f"   Check the 'p2p_music_node_*' folders for saved songs.")


# Example usage and demonstration
async def main():
    """Main function to demonstrate the P2P music generation network"""
    network = P2PMusicNetwork()
    await network.simulate_network_activity()


if __name__ == "__main__":
    asyncio.run(main())