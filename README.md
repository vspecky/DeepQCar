# Self Driving Car Model
A self driving car model that teaches itself using Double Deep Q Learning.  
You can download the code and train your own model by tweaking the various constants and hyperparameters in consts_and_hyperparams.py  
Instructions given below.  
I'm planning to add a lot more in the future, altho for now, hope ya like this.
----
## Dependencies
1) Tensorflow: 
```
pip install tensorflow
```
2) NumPy:
```
pip install numpy
```
3) Matplotlib:
```
pip install matplotlib
```
4) PIL:
```
pip install Pillow
```
5) Pygame:
```
pip install pygame
```
6) h5py:
```
pip install h5py
```
----
## Instructions
Below are some instructions detailing the operation of the code.  
- Download/Clone the repository  
- Feel free to tweak any constants/hyperparameters in consts_and_hyperparams.py  
- Run the code and follow the prompts.  
- Choosing Train shall start training a new model with the current hyperparameters and will overwrite any pre-existing model with the same filename when saved.  
- Choosing Retraining will load a pre-existing model and further train it, overwriting the pre-existing model when saved.  
- Additionally, during training, the Episode-wise rewards shall be logged in the console and on exiting, an Episodes vs Rewards plot will be shown. You can also choose whether to render the game or not. The pygame window will show up regardless of what you choose (It will simply display a black screen if you choose not to render).
- Choosing Testing will render the game, load a pre-existing model and make the agent play the game with the model making the decisions.  
- Choosing Play will render the game and hand over the control of the agent to the user. This is a good way for the user to just play the game or to evaluate the various player/traffic related constants.  
----
## Rendering Controls
During rendering, the user can interact with the environment in certain ways.  
- Whether rendering or not, the user can either press the `Q` key or the quit (X) button on the pygame window to exit the code. If the code is training/retraining a model, the model is first saved before exiting.  
- During rendering, the user can press the `S` key to toggle a visual representation of the sensors of the car. The sensor arms change color based on the distance of detected cars from the player and the colors can be defined by the user in consts_and_hyperparams.py  
- During Testing, the `A` key can be pressed to log the model predictions to the console.  
- During Playing, the `UP` and `DOWN` Arrow Keys can be used to move the player.
----