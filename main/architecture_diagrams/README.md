# Architecture Diagrams

This directory contains Mermaid diagrams describing the complete architecture of the PSD Optimization System.

## Files

1. **01_system_architecture.mermaid** - Complete system architecture showing all layers and components
2. **02_optimization_workflow.mermaid** - Detailed optimization process workflow
3. **03_visualization_workflow.mermaid** - Visualization and editing workflow
4. **04_data_structure.mermaid** - Data structures and class relationships
5. **05_detailed_dependencies.mermaid** - Detailed module dependencies

## How to Convert to Images

### Option 1: Using Mermaid Live Editor (Online)
1. Go to https://mermaid.live/
2. Copy the content of any `.mermaid` file
3. Paste into the editor
4. Click "Download PNG" or "Download SVG"

### Option 2: Using Mermaid CLI (Command Line)
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG
mmdc -i 01_system_architecture.mermaid -o 01_system_architecture.png

# Convert to SVG
mmdc -i 01_system_architecture.mermaid -o 01_system_architecture.svg
```

### Option 3: Using VS Code Extension
1. Install "Markdown Preview Mermaid Support" extension
2. Open the `.mermaid` file
3. Use the preview feature to export as image

### Option 4: Using Python Script
```python
# Install: pip install mermaid
import mermaid as md
md.to_png('01_system_architecture.mermaid', '01_system_architecture.png')
```

## Diagram Descriptions

### System Architecture
Shows the complete system structure with:
- User Interface Layer (Bokeh GUI)
- Main Execution Layer (Orchestration)
- Optimizer Core Layer (Genetic Algorithm)
- Utilities Layer (Logging)
- External Data (Input/Output)

### Optimization Workflow
Step-by-step process of:
- Configuration
- Data Loading
- Candidate Point Generation
- Graph Building
- Genetic Algorithm Evolution
- Result Saving

### Visualization Workflow
Interactive visualization process:
- Data Loading and Matching
- Plot Creation
- User Editing
- Factor Application
- Result Saving

### Data Structure
Class diagram showing:
- Job data structure
- Candidate Points
- Valid Jumps Graph
- Solution representation
- Population management
- Configuration

### Detailed Dependencies
Module import relationships:
- Direct dependencies
- GUI dependencies
- Optimizer Core dependencies

