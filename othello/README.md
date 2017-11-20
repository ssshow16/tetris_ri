OmegaOthello - A Deep Neural Network based Othello/Reversi AI
-------------------------------------------------------------

I built this bot to learn deep reinforcement learning.
Currently the bot is quite weak, however it does seems to be able to learn concepts like board corner, X/Y point, unbalanced edges.
Still, the bot don't seem to hold a chance against search based EDAX program yet.

**Requirements**
    python 3
    numpy
    theano
    numba

Configure theano with GPU support is recommanded.

**Usage**

  Run `python3 train.py` to start training a model. Hit CTRL-C to stop training and the model will be saved at model.pkl.

  Run `python3 play.py` to play against the bot in a CLI, type "h" for help.

**Model and Algorithm**

  Base algorithm is standard Q-learning with a learnable value function approximator.

  The output of value function is 8 by 8 matrix. Invalid moves were pruned as a post-processing step.

  Value function is trained with SGD with ADAM optimizer.

  During playing, OmegaOthello is augmented with a 3-moves minmax search.

  NN architechture is made of 8-step Convoluitonal-GRU followed by 3 full connected MLP layers. Note convolutional weight is shared in all 8 steps.

  NN takes 4 features as input: black pieces, white pieces, valid moves, constant 1(due to zero padding during convolution, this feature can help edge/corner detection)

  Training is using pure self playing, coupled with epsilon-greedy exploration.
