## Imagination Augmented (ML) Agents


Create an I2A agent which collects all necessary imformation from environment

1. Figure out how to pass current PPO agent into I2A 

## Imagination module


## Rollout strategy Distilled policy
1. Actor Network acting as the rolloutout strategy.

## Environment model
1. Create interface to receive observation and output next observation + reward
2. Implement functionality as a Recurrent NN (what loss function? MSE? Cross Entropy?). Prediction on pixel space
3. Implement Conditioned-($\beta$) Variational Auto Encoder to move away from pixel space
4. Figure out how to train C-$\beta$-VAE from training policies online (How to fit training in our RL loop)

## Imagined trajectory Encoder
1. The rollout encoding strategy is carried on by a Recurrent Convolutional Network.
1. Create an LSTM which encodes **backwards** the imagined trajectories into an *rollout embeddings*
2. Many model architecture can be used here (cf. PlaNet algorithm and papers...).
3. Implement a vanilla approach, with great care when it comes to data representation manipulation, i.e. switching between batched representation and sequence representation.

## Aggregator
1. Create an aggregator which simply concatenates the *rollout embeddings* into an *imagination code*.
2. Use an attention mechanism

## Tying it all together
1. Concatenate imagination code with last layer of the Model Free (ppo) into a fully connected layer which outputs an action and a value function.
2. Compute loss and propagate gradients backwards? (FIGURE OUT)
