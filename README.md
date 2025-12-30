# 🤖 Agentic Search AI System

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Qwen 2.5](https://img.shields.io/badge/Model-Qwen%202.5-green.svg)](https://huggingface.co/Qwen)

An intelligent AI agent that autonomously decides when and how to search the web to answer your questions. Built with **Qwen2.5-7B-Instruct** - a lightweight, open-source model that runs locally without API costs.

## 🌟 Features

- 🧠 **Autonomous Decision-Making** - Agent decides when web search is needed
- 🔍 **Smart Search** - DuckDuckGo integration for real-time information
- 💾 **Memory System** - Stores and references accumulated knowledge
- 🎯 **Dynamic Query Generation** - Creates optimized search queries
- 📊 **Result Evaluation** - Assesses relevance and quality
- 🖥️ **Multiple Interfaces** - Web UI, CLI, and Python API
- 💰 **Zero Cost** - No API keys required, runs completely local
- 🔒 **Privacy First** - All processing happens on your machine

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/yourusername/agentic-search-ai.git
cd agentic-search-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run web interface
streamlit run app.py

# OR use CLI
python cli.py "What is the current Bitcoin price?"
```

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [How It Works](#-how-it-works)
- [Testing](#-testing)
- [Contributing](#-contributing)
- [License](#-license)

## 💻 Installation

### Prerequisites

- Python 3.12+
- 8GB+ RAM (12GB+ recommended)
- Internet connection

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_setup.py
```

## 🎮 Usage

### Web Interface

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Command Line

```bash
# Interactive mode
python cli.py

# Direct query
python cli.py "What are the latest AI breakthroughs?"
```

### Python API

```python
from agent import AgenticSearchAgent

agent = AgenticSearchAgent()
result = agent.process_goal("What is the current Bitcoin price?")
print(result['answer'])
```

## 🏗️ Architecture

The system uses a **hybrid approach** for search decisions:

1. **Heuristic Detection** - Fast keyword-based detection
2. **Model Decision** - LLM fallback for edge cases
3. **DuckDuckGo Search** - Real-time web search
4. **Knowledge Storage** - Memory system for results
5. **Answer Synthesis** - Combines knowledge into coherent response

```
User Query → Decision Engine → Search (if needed) → Memory → Answer
```

## 🔄 How It Works

### Search Decision

**Will Search:**
- ✅ "What is the current price of Bitcoin?"
- ✅ "Latest AI news"
- ✅ "Weather today"

**Won't Search:**
- ❌ "What is 2+2?"
- ❌ "Explain photosynthesis"

### Workflow

```python
1. Analyze query for time-sensitive keywords
2. If needed, generate 1-3 search queries
3. Search DuckDuckGo
4. Extract and store results
5. Synthesize answer using search data
```


## 🧪 Testing

```bash
# Test setup
python test_setup.py

# Test search functionality
python test_search.py

# Run unit tests
pytest tests/ -v
```

## 📊 Performance

| Metric | Performance |
|--------|-------------|
| First Load | 30-60 seconds |
| Query (no search) | 5-10 seconds |
| Query (with search) | 15-30 seconds |
| Memory Usage | 6-8GB RAM |
| Cost | **Free!** |

## 🛠️ Troubleshooting

**Slow loading?**
- First download takes time (~3GB model)
- Use GPU for 5-10x speed improvement

**No search results?**
- Check internet connection
- Verify DuckDuckGo is accessible
- Run `python test_search.py`

**Out of memory?**
- Close other applications
- Need 4GB+ RAM minimum

## 📁 Project Structure

```
agentic-search-ai/
├── agent.py              # Core agent
├── app.py                # Web interface
├── requirements.txt      # Dependencies
└── README.md             # This file
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Qwen Team** for the model
- **Hugging Face** for transformers
- **DuckDuckGo** for search
- **Streamlit** for the UI framework

## 📧 Contact

- Hamdi - (hamdi404.cs@gmail.com)
- LinkedIn: https://www.linkedin.com/in/hamdi-mohammed-a0314b213/

---

**Built with ❤️ using Qwen2.5-1.5B-Instruct**

**Ready to use?** Run `streamlit run app.py` 🚀
