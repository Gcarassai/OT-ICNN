import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from ICNN import Kantorovich_Potential
from ComputeOT import W2_squared, ComputeOT
import optuna

# data generator
def sample_data_gen(dataset, batch_size):
    while True:
        indices = np.random.choice(len(dataset), batch_size)
        yield dataset[indices]

def objective(trial):
    NUM_NEURON = 100
    NUM_LAYERS = trial.suggest_int("NUM_LAYERS",3,5)
    INPUT_DIM = dim
    BATCH_SIZE = 1024
    TOT = 90000
    fraz = trial.suggest_float("fraz",0.001,1)
    ITERS = int(np.sqrt(TOT/fraz))
    N_GENERATOR_ITERS = int(np.sqrt(TOT*fraz))
    LR_F = trial.suggest_float("LR_F",1e-4,1e-1,log=True)
    LR_G = trial.suggest_float("LR_G",1e-4,1e-1,log=True)
    # specify the convex function class
    hidden_size_list = [NUM_NEURON for _ in range(NUM_LAYERS)]
    hidden_size_list.append(1)
    # initialize the parameters
    f_model = Kantorovich_Potential(INPUT_DIM, hidden_size_list) 
    g_model = Kantorovich_Potential(INPUT_DIM, hidden_size_list) 
    ComputeOT_gaussians = ComputeOT(dim, f_model,g_model,LR_F,LR_G)
    ComputeOT_gaussians.learn(X_train, Y_train, LAMBDA, BATCH_SIZE, ITERS, N_GENERATOR_ITERS)
    # plot f losses
    plt.plot(ComputeOT_gaussians.losses_f)
    plt.title('f_losses: NUM_LAYERS: '+ str(NUM_LAYERS) + "- fraz:" + str(fraz) +  "- LR_F:"+  str(LR_F) + "- LR_G:"+ str(LR_G)+ '.png')
    plt.savefig('f_losses: NUM_LAYERS: '+ str(NUM_LAYERS) + "- fraz:" + str(fraz) +  "- LR_F:"+  str(LR_F) + "- LR_G:"+ str(LR_G)+ '.png')
    plt.show()  
    # plot w2
    true_W2_squared = alpha**2/2*dim
    plt.plot(ComputeOT_gaussians.W2s, 'b-o', label='W2 during training')
    plt.hlines(true_W2_squared, 0, len(ComputeOT_gaussians.W2s), linestyles='dashed', color='red', label='True W2')
    plt.xlabel('Epochs')
    plt.ylim(-dim, dim)
    plt.legend()
    plt.title('W2: NUM_LAYERS: '+ str(NUM_LAYERS) + "- fraz:" + str(fraz) +  "- LR_F:"+  str(LR_F) + "- LR_G:"+ str(LR_G)+ '.png')
    plt.savefig('W2: NUM_LAYERS: '+ str(NUM_LAYERS) + "- fraz:" + str(fraz) +  "- LR_F:"+  str(LR_F) + "- LR_G:"+ str(LR_G)+ '.png')
    plt.show()
    return abs(W2_squared(X_test,Y_test,ComputeOT_gaussians.f_model,ComputeOT_gaussians.g_model) - true_W2_squared)

if __name__ == '__main__':
    # problem
    alpha = 1 # 5,10
    dim = 784
    LAMBDA = 10
    dataset_size = 500000
    X_train = torch.normal(mean=0., std=1., size=(dataset_size, dim))
    Y_train = torch.normal(mean=alpha, std=1., size=(dataset_size, dim))
    # Define the test set
    test_size = 10000
    X_test = torch.normal(mean=0, std=1, size=(test_size, dim))
    Y_test = torch.normal(mean=alpha, std=1, size=(test_size, dim))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)