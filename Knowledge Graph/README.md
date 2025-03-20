# Knowledge Graph Implementation

This directory contains an implementation of a knowledge graph system that focuses on extracting and analyzing relationships from images and text, with visualization capabilities.

## Features

### Core Components
- Knowledge Graph Construction
  - Image and text processing
  - Entity-Relation extraction
  - Graph storage in JSON format
  - Context-aware analysis

### Image Analysis
- Image extraction from markdown
- Image context analysis using Qwen-VL model
- Automatic entity and relationship detection
- Base64 image encoding support

### Search & Visualization
- Natural language query processing
- Triple-based relationship search
- Interactive graph visualization
- Relationship pattern matching
- NetworkX-based graph rendering

## Usage

### Basic Example
```python
from knowledge_graph import KnowledgeGraphBuilder
from image_searcher import ImageSearcher

# Initialize the graph builder
builder = KnowledgeGraphBuilder(graph_file="knowledge_graph.json")

# Initialize the searcher
searcher = ImageSearcher(graph_file="knowledge_graph.json")

# Perform a search
query = "What can Skipper provide"
results, matched_relationships = searcher.search_images(query)

# Visualize the results
plt = searcher.visualize_graph(
    relationships=matched_relationships,
    title=f"Search Results for: {query}"
)
plt.savefig("search_results_graph.png")
```

## Directory Structure
```
Knowledge Graph/
├── knowledge_graph.py    # Core graph building and image analysis
├── image_searcher.py     # Search and visualization functionality
├── example_usage.py      # Usage examples and demonstrations
├── requirements.txt      # Project dependencies
└── README.md            # Documentation
```

## Requirements
- OpenAI API compatible service
- spaCy with English language model
- NetworkX for graph visualization
- Matplotlib for rendering
- Requests for API calls
- JSON for data storage

## Key Features
1. **Image Analysis**
   - Automatic extraction of entities and relationships from images
   - Context-aware image processing
   - Support for multiple image formats

2. **Search Capabilities**
   - Natural language query processing
   - Relationship-based search
   - Triple pattern matching

3. **Visualization**
   - Interactive graph visualization
   - Customizable graph layouts
   - Support for large-scale knowledge graphs

## Documentation
Detailed documentation for each component is available in their respective module docstrings. See individual files for specific implementation details and usage examples.

## Notes
- Requires valid API credentials for image analysis
- Supports both English and multilingual content
- Graph data is stored in JSON format for easy manipulation and persistence