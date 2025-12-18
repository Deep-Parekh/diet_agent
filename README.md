# DietVA – Local Diet-Planning LangGraph Agent

A **conversational diet-planning agent** using **LangChain + LangGraph** that runs completely locally.

## Features

- **Local Execution**: Runs on MacBook Pro M4 (24 GB RAM) or free Google Colab
- **Local Tools**:
  - BMR / TDEE calculation using Mifflin-St Jeor equation
  - Food lookup via local FoodData Central (FDC) subset
  - Recipe search over a local SQLite database
  - Unit conversion for kitchen measurements and food-specific weights
  - Web search (DuckDuckGo) for recipes/nutrition when local data is insufficient
- **Conversational**: Interactive chat loop with conversation history
- **Model Swapping**: Easy to swap between different LLMs/SLMs
- **Tool Call Logging**: All tool calls logged to JSONL for debugging and analysis
- **Guardrails**:
  - No medical advice, diagnosis, or treatment recommendations
  - No handling of confidential identifiers
  - Protected system prompt

## Quick Start

### Running Locally

#### 1. Start Ollama (Required for Ollama Backend)

Before running the agent with the Ollama backend, ensure Ollama is running:

```bash
# Start Ollama service
ollama serve

# In a separate terminal, verify Ollama is running
ollama list

# Pull the model you want to use (if not already downloaded)
ollama pull qwen2.5:3b
# or
ollama pull llama3.2
```

**Note**: Keep `ollama serve` running in a terminal while using the agent with the Ollama backend.

#### 2. Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

#### 3. Prepare Data

Run the `diet_data_prep.ipynb` notebook to create sample data files, or download full datasets:

- **FoodData Central**: https://fdc.nal.usda.gov/download-datasets.html
- **Recipe Dataset**: https://github.com/Glorf/recipenlg

#### 4. Run the Agent

Open `diet_agent.ipynb` and:

1. **Run all cells** from the top (imports, config, tools, agent setup)
2. **Run the Gradio launch cell** to start the web chat interface:

```python
# Choose your backend (ollama, openai, or hf_local)
config = DietAgentConfig(backend="ollama", model_id="qwen2.5:3b")
demo = create_gradio_chat(config)
demo.launch()  # Opens http://127.0.0.1:7860 in your browser
```

3. **Chat naturally** in the browser interface!

**Backend options:**
- `"ollama"` + `"qwen2.5:3b"`, `"llama3.2"`, `"mistral"` (requires Ollama running - see step 1)
- `"openai"` + `"gpt-4o-mini"` (requires `OPENAI_API_KEY` in `.env`)
- `"hf_local"` + `"Qwen/Qwen2.5-3B-Instruct"` (runs fully locally)

### Running on Google Colab

#### Special Instructions for Colab

1. **Upload the notebook and data files** to Colab:
   - Upload `diet_agent.ipynb` to Colab
   - Upload the `data/` folder containing `fdc_subset.json` and `recipes.db`

2. **Install dependencies** in the first cell:
   ```python
   !pip install langchain langchain-community langgraph gradio ollama
   ```

3. **For Ollama backend on Colab**, you'll need to install and start Ollama:
   ```python
   # Install Ollama
   !curl -fsSL https://ollama.com/install.sh | sh
   
   # Start Ollama in the background
   !OLLAMA_HOST=0.0.0.0:11434 ollama serve &
   
   # Pull the model
   !ollama pull qwen2.5:3b
   ```

4. **For HuggingFace local backend** (recommended for Colab free tier):
   ```python
   !pip install transformers torch
   config = DietAgentConfig(backend="hf_local", model_id="Qwen/Qwen2.5-3B-Instruct")
   ```

5. **Update file paths** in the notebook to point to uploaded files:
   ```python
   # Adjust paths as needed for Colab
   FDC_PATH = "/content/data/fdc_subset.json"
   RECIPES_DB_PATH = "/content/data/recipes.db"
   ```

6. **Run all cells** and launch Gradio:
   ```python
   demo = create_gradio_chat(config)
   demo.launch(share=True)  # Use share=True to get a public URL
   ```

**Note**: Colab free tier has limited RAM. The `hf_local` backend with smaller models (3B) works best. For Ollama on Colab, you may need Colab Pro for sufficient resources.

## Tools

| Tool | Description |
|------|-------------|
| `bmr_tdee_calculator` | Estimate BMR and TDEE using the Mifflin-St Jeor equation for adults. Requires age, sex, height_cm, and weight_kg. Optionally takes activity_level and goal. This is not medical advice. |
| `food_lookup` | Look up foods from a local FoodData Central subset and return approximate calories and macros per 100 g (or a standard serving). Use this instead of guessing nutritional values. |
| `recipe_search` | Search recipes from a local SQLite database by title and ingredients. Can filter by ingredient keywords and simple dietary restrictions. |
| `unit_convert` | Convert quantities between kitchen units (g, kg, oz, lb, ml, cup, tbsp, tsp) and common food-specific weights (apple, egg, etc.). Use this to normalize inputs to grams/ml before suggesting meals or recipes. |
| `web_search` | Use DuckDuckGo text search to fetch recipe or nutrition info when the local database lacks coverage. Returns only titles, URLs, and snippets (no code). |

## Configuration

The `DietAgentConfig` dataclass controls all agent settings:

```python
@dataclass
class DietAgentConfig:
    backend: Literal["hf_local", "ollama", "openai"] = "hf_local"
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.4
    top_p: float = 0.9
    fdc_path: Path = FDC_PATH
    recipes_db_path: Path = RECIPES_DB_PATH
    max_history_turns: int = 6
```

## Model Swapping

Just change the `backend` and `model_id` in the config:

```python
# Ollama (local, requires ollama serve)
config = DietAgentConfig(backend="ollama", model_id="llama3.2")

# OpenAI API
config = DietAgentConfig(backend="openai", model_id="gpt-4o-mini")

# Local HuggingFace model
config = DietAgentConfig(backend="hf_local", model_id="Qwen/Qwen2.5-3B-Instruct")

# Then launch Gradio
demo = create_gradio_chat(config)
demo.launch()
```
