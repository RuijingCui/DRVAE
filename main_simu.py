from argparse import ArgumentParser
import tqdm
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ieee')
plt.rcParams['text.usetex'] = True
# import matplotlib
# matplotlib.use("TkAgg")
from collections import defaultdict
import re
import torch
from torch.distributions import normal,bernoulli
from torch import optim
from DRVAE import DRVAE
from scipy.stats import sem
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from mise_eval import mise_eval
# set random seeds:
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# Loop for replications
def main(args, reptition, load_path):
    set_seed(reptition)
    train_data = pd.read_csv(load_path + f"/{reptition}" + '/train.txt', header=None, sep=' ')
    train_matrix = torch.from_numpy(train_data.to_numpy()).float()
    test_data = pd.read_csv(load_path + f"/{reptition}" + '/test.txt', header=None, sep=' ')
    test_matrix = torch.from_numpy(test_data.to_numpy()).float()
    gird_data = pd.read_csv(load_path + f"/{reptition}" + '/t_grid.txt', header=None, sep=' ')
    t_grid = torch.from_numpy(gird_data.to_numpy()).float()

    ##
    t_train = train_data.values[:, 0]
    y_train = train_data.values[:, -1]
    x_train = train_data.values[:, 1: -1]
    t_train = torch.from_numpy(t_train).squeeze()
    y_train = torch.from_numpy(y_train).squeeze()
    x_train = torch.from_numpy(x_train)

    ##
    t_test = test_data.values[:, 0]
    y_test = test_data.values[:, -1]
    x_test = test_data.values[:, 1: -1]
    t_test = torch.from_numpy(t_test).squeeze()
    y_test = torch.from_numpy(y_test).squeeze()
    x_test = torch.from_numpy(x_test)

    args.x_dim = x_train.shape[1]
    numbers = re.findall(r'\d+', args.dataset_name)
    num_BINnoise = int(numbers[-1])
    num_CONfeatrue = args.x_dim - num_BINnoise
    contfeats = [i for i in range(num_CONfeatrue)]
    binfeats = [i+len(contfeats) for i in range(num_BINnoise)]

    model = DRVAE(args, contfeats, binfeats)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # set batch size
    M = args.batch ## 100
    n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int( x_train.shape[0] / M), list(range( x_train.shape[0]))  ###epoch=200

    Epoch_objective = []
    
    for epoch in range(n_epoch):
        np.random.shuffle(idx)
        for j in range(n_iter_per_epoch):
            batch = np.random.choice(idx, M)
            x_tr, y_tr, t_tr = x_train[batch],y_train[batch], t_train[batch]
            model(x_tr, t_tr, y_tr)
            objective,loss_dict = model.loss_fc()
            optimizer.zero_grad()
            # Calculate gradients
            objective.backward()
            # Update step
            optimizer.step()
            Epoch_objective.append(objective.cpu().detach().item())

    t_grid_hat = model.ADRF(x_test, t_test)
    model.plot_dis(args.dataset_name,x_test)
    mse =  ((t_grid[1,:] - torch.stack(t_grid_hat)) ** 2).mean().item()

    num_treatments = 1
    mise, dpe, ite = mise_eval.compute_ihdpORnews_eval_metrics(args.dataset_name, x_test, num_treatments, model)
    print(f"reptition {reptition} \t mise :: {mise}\t dpe :: {dpe}\t ite :: {ite}\t mse={mse}")  ##

    if reptition % 2 ==0:
        # plt.plot(Epoch_objective)
        # plt.xlabel("Epoch")
        # plt.ylabel("Objective")
        # plt.show()

        t = t_grid[0, :]
        y = t_grid[1, :]
        y_hat = torch.stack(t_grid_hat)
        plt.scatter(t, y, marker='o', ls='-', c= "#16499d",label='simuTruth',alpha=1, zorder=2, s=10, linewidth=2)
        plt.scatter(t, y_hat, marker='o', ls='--', c= "#ef7d1a",label='DRVAE', linewidth=2,cmap='viridis')
        # plt.title(r'ATE with $\alpha_i$')
        plt.xlabel(r'Continuous treatment with $do(t=\alpha_i)$')
        plt.ylabel('Rseponse')
        plt.legend(loc="upper right")
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in')
        plt.tight_layout()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        plt.show()

    merged_tensor = torch.stack((t_grid[0, :], t_grid[1, :], torch.stack(t_grid_hat)), dim=1)
    t_grid_df = pd.DataFrame(merged_tensor.numpy(), columns=['t', 'y_truth', 'y_hat'])
    # try:
    #     t_grid_df.to_excel("mse_results/" + f"{args.dataset_name}_{reptition}_{mse}.xlsx", index=False)
    # except:
    #     OSError

    return mse, mise, dpe, ite
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-reps', type=int, default=1)
    parser.add_argument('-print_every', type=int, default=10)
    parser.add_argument("--x-dim", default=0, type=int)
    parser.add_argument("--latent-dim-c", default=1, type=int) ## \Delta
    parser.add_argument("--latent-dim-t", default=1, type=int) ## \Gamma
    parser.add_argument("--latent-dim-y", default=1, type=int) ## \Upsilon
    parser.add_argument("--latent-dim-e", default=1, type=int) ## \rm E
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-batch', type=int, default=64)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true", default=False)
    parser.add_argument("--alpha1", default=0.1, type=float, help='Reconstr_x')        ## \alpha
    parser.add_argument("--alpha2", default=1.0, type=float, help='Auxiliary_t')       ## \gamma
    parser.add_argument("--alpha3", default=0.1, type=float, help='Auxiliary_y')       ## \delta
    parser.add_argument("--alpha4", default=1.0, type=float, help='Regularization_c')  ## \lamda
    parser.add_argument("--alpha5", default=1.0, type=float, help='Regularization_t')  ## \lamda
    parser.add_argument("--alpha6", default=1.0, type=float, help='Regularization_y')  ## \lamda
    parser.add_argument("--alpha7", default=1.0, type=float, help='Regularization_e')  ## \lamda
    parser.add_argument("--alpha8", default=1.0, type=float, help='KL_divergence')     ## \beta

    parser.add_argument('--dataset_name', type=str, default="simu1_Noise_5con_2loc_10std_10bin")      ## Simu(1)
    # parser.add_argument('--dataset_name', type=str, default="simu1_Noise_10con_2loc_10std_20bin")
    # parser.add_argument('--dataset_name', type=str, default="simu1_Noise_15con_2loc_10std_30bin")
    # parser.add_argument('--dataset_name', type=str, default="simu1_Noise_20con_2loc_10std_40bin")
    # parser.add_argument('--dataset_name', type=str, default="simu1_Noise_25con_2loc_10std_50bin")
    args = parser.parse_args()

    path = "dataset/"+ args.dataset_name + "/eval"
    mses = []
    mises, dpes, ites = [], [], []
    for i in tqdm.tqdm(range(10), desc=f'reptition 10 times for {args.dataset_name}',bar_format="{l_bar}{bar:10}{r_bar}"):
        mse, mise, dpe, ite = main(args, i, path)
        mses.append(mse)
        mises.append(mise)
        dpes.append(dpe)
        ites.append(ite)  ## i-MSE
    plt.plot(mses)
    plt.show()
    print(f"{args.dataset_name}  10 times mean mse :{np.mean(mses)}+-{np.std(mses)}")
    print(f"{args.dataset_name}  10 times mean mise :{np.mean(mises)}+-{np.std(mises)}")
    print(f"{args.dataset_name}  10 times mean dpe :{np.mean(dpes)}+-{np.std(dpes)}")
    print(f"{args.dataset_name}  10 times mean i-MSE :{np.mean(ites)}+-{np.std(ites)}") ## i-MSE
