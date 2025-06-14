# NapsterML - P2P Music AI Network

A peer-to-peer network for sharing and collaboratively training music generation models - like Napster but for AI music models.

## Features

- **Decentralized Model Sharing**: Share and download music AI models across the network
- **Collaborative Learning Groups**: Join genre-specific groups for federated training
- **Music Generation**: Generate songs using downloaded models
- **Peer Discovery**: Automatic discovery of peers on local network
- **Model Search**: Search models by genre, quality, and keywords

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run a Node**
   ```bash
   python p2p_music_napster.py --username "YourUsername" --port 8000
   ```

## Usage Examples

Start multiple nodes on different ports:

```bash
# Node 1 - Producer
python p2p_music_napster.py --username "StudioProducer" --port 8000

# Node 2 - EDM Enthusiast  
python p2p_music_napster.py --username "EDM_Lover" --port 8001

# Node 3 - Classical Collector
python p2p_music_napster.py --username "ClassicalMaestro" --port 8002
```

## Interactive Menu

- Browse network and discover peers
- Search and download models by genre
- Create or join learning groups
- Generate songs using AI models
- View network statistics

## Architecture

- **P2P Discovery**: Multicast announcements + local network scanning
- **Model Storage**: Distributed across peer nodes
- **Federated Learning**: Collaborative model training in groups
- **MIDI Generation**: AI-powered music creation

Built with TensorFlow, Pretty MIDI, and custom P2P networking.