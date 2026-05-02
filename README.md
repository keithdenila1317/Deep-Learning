VizDoom Behavioral Cloning Agent

### Results
* *`model.py`: Defines the core PyTorch neural network architecture. 
* *`agent.py`: The main inference script used to watch the trained AI play. You can load a specific `.pth` checkpoint file into this script to run the model. 

### Data Collection & Processing
* *`Record_episodes.py`*: Launches the VizDoom environment. Once your session ends, the script replays the episode, downscales the resolution for the neural network, and outputs raw `.lmp` replay files.
* *`Lmp_script.py`*: Script that parses through all the raw `.lmp` recording files and converts them into structured NumPy arrays (paired frames and actions).
* *`expert_dataset.npz` & `novice_dataset.npz`*: The final, processed datasets containing the full NumPy arrays of visual frames and corresponding human actions, ready for training.

### Training Pipelines
* *`Train_expert.py` & `train_novice.py`*: The main training loops for our agents.

### Model Weights (Checkpoints)
* *`.pth` files*: Saved PyTorch model weights. 
  * *`expert_model_best.pth`* and *`novice_model_best.pth`* are the primary, most recent, and highest-performing checkpoints based on our final evaluation metrics.
  * *Additional `.pth` files and older iterations are preserved in separate folders.


