import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import pretty_midi
import glob
import os
import pickle
import random
import json
import hashlib
import time
import threading
import socket
import requests
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass, asdict
import uuid
import zipfile
import tempfile
from concurrent.futures import ThreadPoolExecutor
import flwr as fl

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MusicModel:
    """Represents a music generation model"""
    model_id: str
    name: str
    genre: str
    description: str
    creator: str
    version: str
    quality_score: float
    download_count: int
    file_size: int
    created_at: str
    last_updated: str
    tags: List[str]
    sample_songs: List[str]  # URLs to sample songs
    model_hash: str


@dataclass
class LearningGroup:
    """Represents a collaborative learning group"""
    group_id: str
    name: str
    genre: str
    description: str
    admin: str
    members: List[str]
    max_members: int
    is_public: bool
    training_rounds: int
    model_version: str
    created_at: str
    last_active: str
    entry_requirements: Dict[str, any]  # e.g., minimum data quality, genre match


@dataclass
class PeerInfo:
    """Information about a peer in the network"""
    peer_id: str
    username: str
    ip_address: str
    port: int
    last_seen: str
    genres: List[str]
    models_shared: List[str]
    groups_joined: List[str]
    reputation_score: float
    upload_count: int
    download_count: int


class P2PMusicNetwork:
    """Peer-to-peer network for music AI models"""

    def __init__(self, username: str, port: int = 8000):
        self.username = username
        self.peer_id = str(uuid.uuid4())
        self.port = port
        self.ip_address = self.get_local_ip()

        # Network state
        self.peers: Dict[str, PeerInfo] = {}
        self.models: Dict[str, MusicModel] = {}
        self.groups: Dict[str, LearningGroup] = {}
        self.my_groups: Set[str] = set()

        # Local data
        self.local_models: Dict[str, str] = {}  # model_id -> file_path
        self.local_midi_data: List[np.ndarray] = []
        self.reputation_score = 5.0  # Start with neutral reputation

        # Network components
        self.discovery_thread = None
        self.server_thread = None
        self.running = False

        # DHT-like routing table for decentralization
        self.routing_table: Dict[str, List[str]] = {}  # topic -> [peer_ids]

        logger.info(f"Initialized P2P Music Network - Peer ID: {self.peer_id[:8]}...")

    def start_multicast_listener(self):
        """Listen for multicast announcements from other peers"""

        def listener():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('', 9999))

                while self.running:
                    try:
                        data, addr = sock.recvfrom(1024)
                        announce_data = json.loads(data.decode())

                        if (announce_data.get("type") == "peer_announce"
                                and announce_data.get("peer_id") != self.peer_id):
                            # Discovered a new peer via multicast
                            peer_info = {
                                "peer_id": announce_data["peer_id"],
                                "username": announce_data["username"],
                                "genres": [],
                                "reputation": 5.0
                            }

                            self.add_peer(peer_info, announce_data["ip"], announce_data["port"])

                    except socket.timeout:
                        continue
                    except Exception as e:
                        logger.debug(f"Multicast listener error: {e}")

            except Exception as e:
                logger.error(f"Failed to start multicast listener: {e}")

        # Start listener in background thread
        listener_thread = threading.Thread(target=listener)
        listener_thread.daemon = True
        listener_thread.start()

    def get_local_ip(self):
        """Get local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"

    def start_network(self):
        """Start the P2P network"""
        self.running = True

        # Start HTTP server for peer communication
        self.server_thread = threading.Thread(target=self.start_http_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        # Start multicast listener - THIS WAS MISSING!
        self.start_multicast_listener()

        # Start peer discovery
        self.discovery_thread = threading.Thread(target=self.peer_discovery_loop)
        self.discovery_thread.daemon = True
        self.discovery_thread.start()

        logger.info(f"🎵 P2P Music Network started on {self.ip_address}:{self.port}")
        logger.info(f"👤 Username: {self.username}")
        logger.info(f"🆔 Peer ID: {self.peer_id[:8]}...")

    def start_http_server(self):
        """Start HTTP server for peer communication"""
        from http.server import HTTPServer, BaseHTTPRequestHandler

        class P2PHandler(BaseHTTPRequestHandler):
            def __init__(self, p2p_network, *args, **kwargs):
                self.p2p_network = p2p_network
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self.path == "/peer_info":
                    self.send_peer_info()
                elif self.path == "/peer_list":
                    self.send_peer_list()
                elif self.path.startswith("/models"):
                    self.send_models_list()
                elif self.path.startswith("/groups"):
                    self.send_groups_list()
                elif self.path.startswith("/download/"):
                    self.handle_download()
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path == "/join_group":
                    self.handle_join_group()
                elif self.path == "/create_group":
                    self.handle_create_group()
                elif self.path == "/share_model":
                    self.handle_share_model()
                elif self.path == "/group_announce":
                    self.handle_group_announce()
                elif self.path == "/group_notification":
                    self.handle_group_notification()
                else:
                    self.send_error(404)

            def send_peer_info(self):
                peer_info = {
                    "peer_id": self.p2p_network.peer_id,
                    "username": self.p2p_network.username,
                    "genres": list(set([model.genre for model in self.p2p_network.models.values()])),
                    "models_count": len(self.p2p_network.local_models),
                    "groups": list(self.p2p_network.my_groups),
                    "reputation": self.p2p_network.reputation_score
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(peer_info).encode())

            def send_peer_list(self):
                """Send list of known peers"""
                peers_data = []
                for peer in self.p2p_network.peers.values():
                    peers_data.append({
                        "peer_id": peer.peer_id,
                        "username": peer.username,
                        "ip_address": peer.ip_address,
                        "port": peer.port,
                        "genres": peer.genres,
                        "reputation": peer.reputation_score
                    })

                response_data = {"peers": peers_data}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())

            def send_models_list(self):
                """Send list of available models"""
                models_data = []
                for model in self.p2p_network.models.values():
                    models_data.append(asdict(model))

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(models_data).encode())

            def send_groups_list(self):
                """Send list of available groups"""
                groups_data = []
                for group in self.p2p_network.groups.values():
                    if group.is_public:
                        groups_data.append(asdict(group))

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(groups_data).encode())

            def handle_group_announce(self):
                """Handle group announcement from another peer"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    group_data = json.loads(post_data.decode('utf-8'))

                    # Add the group to our known groups
                    group = LearningGroup(
                        group_id=group_data["group_id"],
                        name=group_data["name"],
                        genre=group_data["genre"],
                        description=group_data["description"],
                        admin=group_data["admin"],
                        members=[group_data["admin"]],
                        max_members=group_data["max_members"],
                        is_public=group_data["is_public"],
                        training_rounds=0,
                        model_version="1.0.0",
                        created_at=group_data["created_at"],
                        last_active=group_data["timestamp"],
                        entry_requirements={}
                    )

                    self.p2p_network.groups[group.group_id] = group

                    self.send_response(200)
                    self.end_headers()

                except Exception as e:
                    logger.error(f"Failed to handle group announcement: {e}")
                    self.send_error(500)

            def handle_group_notification(self):
                """Handle group join notifications"""
                try:
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    notification = json.loads(post_data.decode('utf-8'))

                    logger.info(f"👥 {notification['username']} joined group {notification['group_id'][:8]}...")

                    self.send_response(200)
                    self.end_headers()

                except Exception as e:
                    logger.error(f"Failed to handle group notification: {e}")
                    self.send_error(500)

            def handle_download(self):
                """Handle model download requests"""
                # Placeholder implementation
                self.send_error(501, "Download not implemented yet")

            def handle_join_group(self):
                """Handle group join requests"""
                # Placeholder implementation
                self.send_error(501, "Join group not implemented yet")

            def handle_create_group(self):
                """Handle group creation requests"""
                # Placeholder implementation
                self.send_error(501, "Create group not implemented yet")

            def handle_share_model(self):
                """Handle model sharing requests"""
                # Placeholder implementation
                self.send_error(501, "Share model not implemented yet")

        # Create server with closure to pass p2p_network
        handler = lambda *args, **kwargs: P2PHandler(self, *args, **kwargs)

        try:
            server = HTTPServer(("", self.port), handler)
            server.serve_forever()
        except Exception as e:
            logger.error(f"HTTP server error: {e}")

    def peer_discovery_loop(self):
        """Discover peers in the network"""
        # Initial rapid discovery
        for i in range(3):
            try:
                self.discover_peers()
                time.sleep(2)  # Quick initial scans
            except Exception as e:
                logger.error(f"Initial peer discovery error: {e}")

        # Then normal discovery loop
        while self.running:
            try:
                self.discover_peers()
                self.cleanup_inactive_peers()
                time.sleep(15)  # Reduced from 30 seconds
            except Exception as e:
                logger.error(f"Peer discovery error: {e}")

    def discover_peers(self):
        """Discover peers using multicast and known peers"""
        # Multicast discovery
        self.multicast_announce()

        # Scan local network
        self.scan_local_network()

        # Ask known peers for their peer lists
        self.request_peer_lists()

    def multicast_announce(self):
        """Announce presence via multicast"""
        try:
            # Try both broadcast and multicast
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

            announce_data = {
                "type": "peer_announce",
                "peer_id": self.peer_id,
                "username": self.username,
                "ip": self.ip_address,
                "port": self.port,
                "timestamp": datetime.now().isoformat()
            }

            message = json.dumps(announce_data).encode()

            # Broadcast
            sock.sendto(message, ('255.255.255.255', 9999))

            # Also try localhost for same-machine testing
            sock.sendto(message, ('127.0.0.1', 9999))

            sock.close()
            logger.debug(f"Sent multicast announcement")

        except Exception as e:
            logger.debug(f"Multicast announce failed: {e}")

    def scan_local_network(self):
        """Scan local network for peers"""
        base_ip = ".".join(self.ip_address.split(".")[:-1])

        def check_peer(ip, port):
            try:
                # Try multiple common ports, not just self.port
                for p in [8000, 8001, 8002, 8003, 8004, 8005]:
                    try:
                        response = requests.get(f"http://{ip}:{p}/peer_info", timeout=1)
                        if response.status_code == 200:
                            peer_data = response.json()
                            self.add_peer(peer_data, ip, p)
                            return
                    except:
                        continue
            except:
                pass

        # Scan local network range
        with ThreadPoolExecutor(max_workers=50) as executor:
            # Scan localhost first (for testing on same machine)
            for port in range(8000, 8010):
                if port != self.port:
                    executor.submit(check_peer, "127.0.0.1", port)

            # Then scan local network
            if not self.ip_address.startswith("127."):
                for i in range(1, 255):
                    ip = f"{base_ip}.{i}"
                    if ip != self.ip_address:
                        executor.submit(check_peer, ip, None)

    def add_peer(self, peer_data: dict, ip: str, port: int):
        """Add a discovered peer"""
        # Don't add ourselves
        if peer_data["peer_id"] == self.peer_id:
            return

        peer_info = PeerInfo(
            peer_id=peer_data["peer_id"],
            username=peer_data["username"],
            ip_address=ip,
            port=port,
            last_seen=datetime.now().isoformat(),
            genres=peer_data.get("genres", []),
            models_shared=peer_data.get("models", []),
            groups_joined=peer_data.get("groups", []),
            reputation_score=peer_data.get("reputation", 5.0),
            upload_count=0,
            download_count=0
        )

        if peer_info.peer_id not in self.peers:
            self.peers[peer_info.peer_id] = peer_info
            logger.info(f"📡 NEW PEER: {peer_info.username} @ {ip}:{port} ({peer_info.peer_id[:8]}...)")
        else:
            # Update existing peer
            self.peers[peer_info.peer_id] = peer_info
            logger.debug(f"📡 Updated peer: {peer_info.username}")

    def create_learning_group(self, name: str, genre: str, description: str,
                              max_members: int = 10, is_public: bool = True,
                              entry_requirements: Dict = None) -> str:
        """Create a new learning group"""
        group_id = str(uuid.uuid4())

        group = LearningGroup(
            group_id=group_id,
            name=name,
            genre=genre,
            description=description,
            admin=self.peer_id,
            members=[self.peer_id],
            max_members=max_members,
            is_public=is_public,
            training_rounds=0,
            model_version="1.0.0",
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            entry_requirements=entry_requirements or {}
        )

        self.groups[group_id] = group
        self.my_groups.add(group_id)

        # Announce group to network
        self.announce_group(group)

        logger.info(f"🎯 Created learning group: {name} ({genre})")
        return group_id

    def join_learning_group(self, group_id: str) -> bool:
        """Join an existing learning group"""
        if group_id not in self.groups:
            logger.error(f"Group {group_id} not found")
            return False

        group = self.groups[group_id]

        # Check if group is full
        if len(group.members) >= group.max_members:
            logger.error(f"Group {group.name} is full")
            return False

        # Check entry requirements
        if not self.meets_requirements(group.entry_requirements):
            logger.error(f"Don't meet requirements for group {group.name}")
            return False

        # Join group
        group.members.append(self.peer_id)
        group.last_active = datetime.now().isoformat()
        self.my_groups.add(group_id)

        # Notify group admin
        self.notify_group_join(group_id)

        logger.info(f"🎵 Joined learning group: {group.name}")
        return True

    def leave_learning_group(self, group_id: str):
        """Leave a learning group"""
        if group_id in self.groups:
            group = self.groups[group_id]
            if self.peer_id in group.members:
                group.members.remove(self.peer_id)
                self.my_groups.discard(group_id)
                logger.info(f"👋 Left learning group: {group.name}")

    def switch_learning_group(self, from_group_id: str, to_group_id: str) -> bool:
        """Switch from one learning group to another"""
        if self.join_learning_group(to_group_id):
            self.leave_learning_group(from_group_id)
            logger.info(f"🔄 Switched from group {from_group_id[:8]}... to {to_group_id[:8]}...")
            return True
        return False

    def discover_learning_groups(self, genre_filter: str = None) -> List[LearningGroup]:
        """Discover available learning groups"""
        discovered_groups = []

        # Ask all known peers for their groups
        for peer in self.peers.values():
            try:
                response = requests.get(f"http://{peer.ip_address}:{peer.port}/groups", timeout=5)
                if response.status_code == 200:
                    groups_data = response.json()
                    for group_data in groups_data:
                        group = LearningGroup(**group_data)
                        if genre_filter is None or group.genre.lower() == genre_filter.lower():
                            discovered_groups.append(group)
                            self.groups[group.group_id] = group
            except Exception as e:
                logger.debug(f"Failed to get groups from {peer.username}: {e}")

        return discovered_groups

    def start_federated_training(self, group_id: str, rounds: int = 10):
        """Start federated training for a group"""
        if group_id not in self.my_groups:
            logger.error(f"Not a member of group {group_id}")
            return

        group = self.groups[group_id]

        if group.admin == self.peer_id:
            # I'm the admin, start as server
            self.start_group_training_server(group_id, rounds)
        else:
            # Join as client
            self.start_group_training_client(group_id)

    def search_models(self, query: str = "", genre: str = "", min_quality: float = 0.0) -> List[MusicModel]:
        """Search for music models across the network"""
        results = []

        # Search local models
        for model in self.models.values():
            if self.matches_search(model, query, genre, min_quality):
                results.append(model)

        # Search peer models
        for peer in self.peers.values():
            try:
                response = requests.get(f"http://{peer.ip_address}:{peer.port}/models", timeout=5)
                if response.status_code == 200:
                    models_data = response.json()
                    for model_data in models_data:
                        model = MusicModel(**model_data)
                        if self.matches_search(model, query, genre, min_quality):
                            results.append(model)
            except Exception as e:
                logger.debug(f"Failed to search models from {peer.username}: {e}")

        # Sort by quality and popularity
        results.sort(key=lambda m: (m.quality_score, m.download_count), reverse=True)
        return results

    def matches_search(self, model: MusicModel, query: str, genre: str, min_quality: float) -> bool:
        """Check if model matches search criteria"""
        if min_quality > 0 and model.quality_score < min_quality:
            return False

        if genre and model.genre.lower() != genre.lower():
            return False

        if query:
            search_text = f"{model.name} {model.description} {' '.join(model.tags)}".lower()
            if query.lower() not in search_text:
                return False

        return True

    def download_model(self, model_id: str, save_path: str = None) -> bool:
        """Download a model from the network"""
        # Find which peer has the model
        model_peer = None
        for peer in self.peers.values():
            if model_id in peer.models_shared:
                model_peer = peer
                break

        if not model_peer:
            logger.error(f"Model {model_id} not found in network")
            return False

        try:
            # Download model
            download_url = f"http://{model_peer.ip_address}:{model_peer.port}/download/{model_id}"
            response = requests.get(download_url, stream=True, timeout=30)

            if response.status_code == 200:
                if save_path is None:
                    save_path = f"downloaded_model_{model_id[:8]}.zip"

                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Update download count
                if model_id in self.models:
                    self.models[model_id].download_count += 1

                logger.info(f"📥 Downloaded model {model_id[:8]}... to {save_path}")
                return True
            else:
                logger.error(f"Failed to download model: HTTP {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False

    def generate_song_from_model(self, model_id: str, length: int = 200,
                                 temperature: float = 0.8, output_file: str = None) -> str:
        """Generate a song using a specific model"""
        if model_id not in self.local_models:
            logger.error(f"Model {model_id} not available locally. Download it first.")
            return None

        try:
            # Load the model
            model_path = self.local_models[model_id]
            generator = self.load_model_from_file(model_path)

            # Generate music
            generated_music = generator.generate_music(length=length, temperature=temperature)

            # Save as MIDI
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"generated_song_{model_id[:8]}_{timestamp}.mid"

            generator.piano_roll_to_midi(generated_music, output_file)

            logger.info(f"🎵 Generated song: {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Song generation failed: {e}")
            return None

    def get_network_stats(self) -> Dict:
        """Get network statistics"""
        return {
            "total_peers": len(self.peers),
            "total_models": len(self.models),
            "total_groups": len(self.groups),
            "my_groups": len(self.my_groups),
            "my_models": len(self.local_models),
            "reputation": self.reputation_score,
            "genres_available": list(set([model.genre for model in self.models.values()])),
            "top_models": sorted(self.models.values(), key=lambda m: m.quality_score, reverse=True)[:5]
        }

    def display_network_browser(self):
        """Display network browser interface"""
        print("\n" + "=" * 80)
        print("🎵 NAPSTER_ML - P2P MUSIC AI NETWORK BROWSER")
        print("=" * 80)

        stats = self.get_network_stats()

        print(f"👥 Connected Peers: {stats['total_peers']}")
        print(f"🤖 Available Models: {stats['total_models']}")
        print(f"🎯 Learning Groups: {stats['total_groups']}")
        print(f"⭐ Your Reputation: {stats['reputation']:.1f}")

        print(f"\n🎼 Available Genres: {', '.join(stats['genres_available'])}")

        print(f"\n🏆 Top Models:")
        for i, model in enumerate(stats['top_models'], 1):
            print(f"  {i}. {model.name} ({model.genre}) - Quality: {model.quality_score:.1f}")

        print(f"\n🎯 Your Groups:")
        for group_id in self.my_groups:
            if group_id in self.groups:
                group = self.groups[group_id]
                print(f"  • {group.name} ({group.genre}) - {len(group.members)} members")

    def cleanup_inactive_peers(self):
        """Remove inactive peers"""
        current_time = datetime.now()
        inactive_peers = []

        for peer_id, peer in self.peers.items():
            last_seen = datetime.fromisoformat(peer.last_seen)
            if current_time - last_seen > timedelta(minutes=10):
                inactive_peers.append(peer_id)

        for peer_id in inactive_peers:
            del self.peers[peer_id]
            logger.debug(f"Removed inactive peer: {peer_id[:8]}...")

    def shutdown(self):
        """Shutdown the network"""
        self.running = False
        logger.info("🛑 Shutting down NapsterML - P2P Music Network")

    def announce_group(self, group: LearningGroup):
        """Announce a new group to the network"""
        try:
            group_announcement = {
                "type": "group_announce",
                "group_id": group.group_id,
                "name": group.name,
                "genre": group.genre,
                "description": group.description,
                "admin": group.admin,
                "max_members": group.max_members,
                "is_public": group.is_public,
                "created_at": group.created_at,
                "peer_id": self.peer_id,
                "timestamp": datetime.now().isoformat()
            }

            # Announce to all known peers
            for peer in self.peers.values():
                try:
                    requests.post(
                        f"http://{peer.ip_address}:{peer.port}/group_announce",
                        json=group_announcement,
                        timeout=5
                    )
                except Exception as e:
                    logger.debug(f"Failed to announce group to {peer.username}: {e}")

        except Exception as e:
            logger.error(f"Group announcement failed: {e}")

    def request_peer_lists(self):
        """Request peer lists from known peers to discover more peers"""
        for peer in list(self.peers.values()):
            try:
                response = requests.get(
                    f"http://{peer.ip_address}:{peer.port}/peer_list",
                    timeout=5
                )
                if response.status_code == 200:
                    peer_list = response.json()
                    for peer_data in peer_list.get("peers", []):
                        # Add new peers we don't know about
                        if peer_data["peer_id"] not in self.peers and peer_data["peer_id"] != self.peer_id:
                            self.add_peer(peer_data, peer_data["ip_address"], peer_data["port"])
            except Exception as e:
                logger.debug(f"Failed to get peer list from {peer.username}: {e}")

    def meets_requirements(self, requirements: Dict) -> bool:
        """Check if this peer meets the group entry requirements"""
        # Simple implementation - can be extended
        if not requirements:
            return True

        # Check minimum reputation if required
        min_reputation = requirements.get("min_reputation", 0)
        if self.reputation_score < min_reputation:
            return False

        # Check required genres if specified
        required_genres = requirements.get("required_genres", [])
        if required_genres:
            my_genres = list(set([model.genre for model in self.models.values()]))
            if not any(genre in my_genres for genre in required_genres):
                return False

        return True

    def notify_group_join(self, group_id: str):
        """Notify group admin about a new member joining"""
        if group_id not in self.groups:
            return

        group = self.groups[group_id]
        admin_peer = self.peers.get(group.admin)

        if admin_peer:
            try:
                notification = {
                    "type": "group_join",
                    "group_id": group_id,
                    "new_member": self.peer_id,
                    "username": self.username,
                    "timestamp": datetime.now().isoformat()
                }

                requests.post(
                    f"http://{admin_peer.ip_address}:{admin_peer.port}/group_notification",
                    json=notification,
                    timeout=5
                )
            except Exception as e:
                logger.debug(f"Failed to notify group admin: {e}")

    def start_group_training_server(self, group_id: str, rounds: int):
        """Start federated training as server (group admin)"""
        logger.info(f"🚀 Starting federated training server for group {group_id}")
        # Placeholder for federated learning server implementation
        # This would use the flwr framework to coordinate training
        pass

    def start_group_training_client(self, group_id: str):
        """Join federated training as client"""
        logger.info(f"🤝 Joining federated training for group {group_id}")
        # Placeholder for federated learning client implementation
        # This would use the flwr framework to participate in training
        pass

    def load_model_from_file(self, model_path: str):
        """Load a model from file"""
        # Simple implementation - returns a basic generator
        generator = SimpleMIDIGenerator()
        generator.build_model()
        return generator


# Simplified MIDI generator for the prototype
class SimpleMIDIGenerator:
    def __init__(self, sequence_length=64, fs=16):
        self.sequence_length = sequence_length
        self.fs = fs
        self.model = None

    def build_model(self):
        """Build a simple LSTM model"""
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 16)),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            Dense(16, activation='sigmoid')
        ])

        self.model.compile(optimizer=Adam(0.001), loss='mse')
        return self.model

    def generate_music(self, length=200, temperature=0.8):
        """Generate music (simplified for prototype)"""
        if self.model is None:
            self.build_model()

        # Generate random music for prototype
        return np.random.random((length, 16))

    def piano_roll_to_midi(self, piano_roll, filename):
        """Convert to MIDI (simplified)"""
        # Create a simple MIDI file for prototype
        pm = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        for i, frame in enumerate(piano_roll[:100]):  # Limit for prototype
            if frame[0] > 0.3:  # Simple threshold
                note = pretty_midi.Note(
                    velocity=int(frame[1] * 127),
                    pitch=int(frame[0] * 88) + 20,
                    start=i * 0.25,
                    end=(i + 1) * 0.25
                )
                instrument.notes.append(note)

        pm.instruments.append(instrument)
        pm.write(filename)
        return filename


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Napster ML - P2P Music AI Network. Napster for Song ML Models")
    parser.add_argument("--username", type=str, required=True, help="Your username")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--mode", choices=["interactive", "daemon"], default="interactive")

    args = parser.parse_args()

    # Initialize network
    network = P2PMusicNetwork(args.username, args.port)
    network.start_network()

    if args.mode == "interactive":
        # Interactive mode
        print(f"\n🎵 Welcome to NapsterML - the P2P Music AI Network!")
        print(f"Username: {args.username}")
        print(f"Peer ID: {network.peer_id[:8]}...")

        while True:
            try:
                print("\n📋 MENU:")
                print("1. Browse Network")
                print("2. Search Models")
                print("3. Create Learning Group")
                print("4. Join Learning Group")
                print("5. Discover Groups")
                print("6. Generate Song")
                print("7. Network Stats")
                print("8. Exit")

                choice = input("\nEnter choice (1-8): ").strip()

                if choice == "1":
                    network.display_network_browser()

                elif choice == "2":
                    query = input("Search query (or Enter for all): ").strip()
                    genre = input("Genre filter (or Enter for all): ").strip()

                    models = network.search_models(query, genre)
                    print(f"\n🔍 Found {len(models)} models:")
                    for i, model in enumerate(models[:10], 1):
                        print(f"  {i}. {model.name} ({model.genre}) - Quality: {model.quality_score:.1f}")
                        print(f"     📝 {model.description}")
                        print(f"     👤 By: {model.creator}")
                        print(f"     📥 Downloads: {model.download_count}")

                elif choice == "3":
                    name = input("Group name: ").strip()
                    genre = input("Genre: ").strip()
                    description = input("Description: ").strip()
                    max_members = int(input("Max members (default 10): ") or "10")

                    group_id = network.create_learning_group(name, genre, description, max_members)
                    print(f"✅ Created group: {name} (ID: {group_id[:8]}...)")

                elif choice == "4":
                    groups = network.discover_learning_groups()
                    if groups:
                        print("\n🎯 Available Groups:")
                        for i, group in enumerate(groups, 1):
                            print(
                                f"  {i}. {group.name} ({group.genre}) - {len(group.members)}/{group.max_members} members")
                            print(f"     📝 {group.description}")

                        try:
                            selection = int(input("Select group number (0 to cancel): "))
                            if 1 <= selection <= len(groups):
                                selected_group = groups[selection - 1]
                                if network.join_learning_group(selected_group.group_id):
                                    print(f"✅ Joined group: {selected_group.name}")
                                else:
                                    print("❌ Failed to join group")
                        except ValueError:
                            print("Invalid selection")
                    else:
                        print("No groups found")

                elif choice == "5":
                    genre_filter = input("Genre filter (or Enter for all): ").strip()
                    groups = network.discover_learning_groups(genre_filter if genre_filter else None)
                    print(f"\n🔍 Discovered {len(groups)} groups:")
                    for group in groups:
                        print(f"  • {group.name} ({group.genre}) - {len(group.members)} members")
                        print(f"    📝 {group.description}")
                        print(f"    👤 Admin: {group.admin[:8]}...")

                elif choice == "6":
                    # For prototype, create a simple generated song
                    generator = SimpleMIDIGenerator()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"p2p_generated_song_{timestamp}.mid"

                    print("🎵 Generating song...")
                    music = generator.generate_music(length=100)
                    generator.piano_roll_to_midi(music, filename)
                    print(f"✅ Generated: {filename}")

                elif choice == "7":
                    network.display_network_browser()

                elif choice == "8":
                    break

                else:
                    print("Invalid choice")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {e}")

        network.shutdown()

    else:
        # Daemon mode
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            network.shutdown()


if __name__ == "__main__":
    main()

# Example usage scenarios:
"""
# Start different nodes:

# Music Producer Node
python p2p_music_napster.py --username "StudioA_Producer" --port 8000

# Electronic Music Enthusiast
python p2p_music_napster.py --username "EDM_Lover" --port 8001

# Classical Music Collector
python p2p_music_napster.py --username "ClassicalMaestro" --port 8002

# Jazz Musician
python p2p_music_napster.py --username "JazzCat" --port 8003

# Example distributed network:
# Node 1: Creates "Epic Orchestral" learning group
# Node 2: Joins group and contributes classical MIDI files
# Node 3: Searches for "epic" models and downloads favorites
# Node 4: Creates competing "Jazz Fusion" group
# Node 2: Switches from orchestral to jazz group
# All nodes: Collaborate on training, share models, generate songs
"""