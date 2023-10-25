# Natural Chess Processing

Repository for the Generative Artificial Intelligence Fall 2023 course. Explores the advantages and disadvantages of applying using natural language processing techniques to chess games.

# Repository Structure

- `src` - Source code directory.
    - `src/data` - Module containing code related to data processing.
    - `src/models` - Module containing code related to neural networks.
    - `src/training_loop` - Module containing code related to training the networks.
- `config_files` - Configuration files for training.
- `scripts` - Utility scripts for downloading and processing chess game databases.
- `models` - Pretrained model checkpoints.
- `train.py` - Training script.
- `engine.py` - UCI engine script.

# How to Run
0. *(Optional)* Create a virtual environment
    1. Create the virtual environment
    ```bash
    python -m venv .venv
    ``` 
    2. Activate the new environment
    ```bash
    source ./.venv/bin/activate
    ```
1. Install requirements from `requirements.txt`
```bash
python -m pip install -r requirements.txt
```
2. Make sure the wrapping script (`engine_wrap.sh`) is correct:
    - Default script activates virtual environment in `./.venv`. To use a different environment, use a different command, i.e. `conda activate env`.
3. Launch a chess engine GUI.
    - There are a wide selections of chess GUIs, but the engine script has been primarily tested with the [Nibbler](https://github.com/rooklift/nibbler) GUI, available for Windows and Linux.
4. Load the `engine_wrap.sh` script as a chess engine.

# How to Train

1. Install requirements from `requirements.txt`
```bash
python -m pip install -r requirements.txt
```

2. Modify/create the config file
    - Examples are available in `config_files/`

3. Run `train.py`
    - Example command: 
    ```bash
    python train.py config_files/position_baseline.toml
    ```
    Run 
    ```bash
    python train.py --help
    ```
    to view all arguments.