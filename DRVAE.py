from torch import nn
import torch
import torch.nn.functional as F
from distributions_forRep import *
from torch.distributions import bernoulli, normal
from collections import defaultdict
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt
class DRVAE(nn.Module):
    def __init__(self, args, contfeats=None, binfeats=None):
        super(DRVAE, self).__init__()
        self.contfeats = contfeats
        self.binfeats = binfeats
        self.args = args

        ## zc ~ p(zc|x)
        self.q_zc_x_dist = q_z_x(self.args.x_dim, self.args.hidden_dim, self.args.num_layers, self.args.latent_dim_c)
        self.q_zt_x_dist = q_z_x(self.args.x_dim, self.args.hidden_dim, self.args.num_layers, self.args.latent_dim_t)
        self.q_zy_x_dist = q_z_x(self.args.x_dim, self.args.hidden_dim, self.args.num_layers, self.args.latent_dim_y)
        self.q_ze_x_dist = q_z_x(self.args.x_dim, self.args.hidden_dim, self.args.num_layers, self.args.latent_dim_e)

        ### p(z)~N(0,1)
        self.p_zt_dist = normal.Normal(torch.zeros(self.args.latent_dim_t), torch.ones(self.args.latent_dim_t))  ## =N01  zt
        self.p_zc_dist = normal.Normal(torch.zeros(self.args.latent_dim_c), torch.ones(self.args.latent_dim_c))  ##zc
        self.p_zy_dist = normal.Normal(torch.zeros(self.args.latent_dim_y), torch.ones(self.args.latent_dim_y))  ##zy
        self.p_ze_dist = normal.Normal(torch.zeros(self.args.latent_dim_e), torch.ones(self.args.latent_dim_e))  ##ze
        
        if self.binfeats!= None:
            ## x_con ~ p(x|zc,zt,zy,ze)
            self.p_x_z_dist = p_x_zi_zc_za_ze([self.args.latent_dim_c + self.args.latent_dim_t + self.args.latent_dim_y + self.args.latent_dim_e] + [self.args.hidden_dim] * self.args.num_layers + [len(self.contfeats)])
            ## x_bin ~ p(x|zc,zt,zy,ze)
            self.p_xbin_z_dist = p_xbin_zi_zc_za_ze([self.args.latent_dim_c + self.args.latent_dim_t + self.args.latent_dim_y + self.args.latent_dim_e] +  [self.args.hidden_dim] * self.args.num_layers + [len(self.binfeats)])
        else:
            self.p_x_z_dist = p_x_zi_zc_za_ze([self.args.latent_dim_c + self.args.latent_dim_t + self.args.latent_dim_y + self.args.latent_dim_e] + [self.args.hidden_dim] * self.args.num_layers + [self.args.x_dim])
        
        ## t ~ p(t|zc,zt) # treatment
        self.p_t_zc_zt_dist = p_t_zc_zt([self.args.latent_dim_c + self.args.latent_dim_t] + [self.args.hidden_dim] * self.args.num_layers)
        ## y ~ p(y|t,zc,zy) # outcome
        self.p_y_t_zc_zy_dist = p_y_t_zc_zy([self.args.latent_dim_c + self.args.latent_dim_y + 1] + [self.args.hidden_dim] * self.args.num_layers)

    def forward(self, x, t, y):
        """
        :param x,t,y: the input of training model
        :return: the result of training model
        """
        self.x_train = x
        self.t_train = t
        self.y_train = y

        # encoder
        self.encode(self.x_train)
        
        # reparameterization
        self.reparameterization()

        # decoder
        self.decode()
        
        # return mu_logvar_of_zc, mu_logvar_of_zt, mu_logvar_of_zy, mu_logvar_of_ze

    def encode(self,x_train):
        """
        encoding part
        :param x: input x_train
        :return: mu and log_var
        """
        # inferred distribution over z
        # z_infer = q_z_tyx_dist(xy=xy, t=t_train) ## torch.Size([100, 20])
        self.mu_var_of_zc = self.q_zc_x_dist(x_train)
        self.zc_infer = normal.Normal(*self.mu_var_of_zc)

        self.mu_var_of_zt = self.q_zt_x_dist(x_train)
        self.zt_infer = normal.Normal(*self.mu_var_of_zt)

        self.mu_var_of_zy = self.q_zy_x_dist(x_train)
        self.zy_infer = normal.Normal(*self.mu_var_of_zy)

        self.mu_var_of_ze = self.q_ze_x_dist(x_train)
        self.ze_infer = normal.Normal(*self.mu_var_of_ze)

        # return self.mu_logvar_of_zc, self.mu_logvar_of_zt, self.mu_logvar_of_zy, self.mu_logvar_of_ze

    def reparameterization(self):
        """
        Given a standard gaussian distribution epsilon ~ N(0,1),
        we can sample the random variable z as per z = mu + sigma * epsilon
        :param mu:
        :param log_var:
        :return: sampled z
        """
        self.mu_zc, self.var_zc = self.mu_var_of_zc
        eps = torch.randn_like(self.var_zc)
        self.zc_infer_sample = self.mu_zc + self.var_zc * eps

        self.mu_zt, self.var_zt = self.mu_var_of_zt
        eps = torch.randn_like(self.var_zt)
        self.zt_infer_sample = self.mu_zt + self.var_zt * eps

        self.mu_zy, self.var_zy = self.mu_var_of_zy
        eps = torch.randn_like(self.var_zy)
        self.zy_infer_sample = self.mu_zy + self.var_zy * eps

        self.mu_ze, self.var_ze = self.mu_var_of_ze
        eps = torch.randn_like(self.var_ze)
        self.ze_infer_sample = self.mu_ze + self.var_ze * eps

        # return self.zc_infer_sample, self.zt_infer_sample, self.zy_infer_sample, self.ze_infer_sample

    def decode(self):
        # p(x|zc,zt,zy,ze)
        if self.binfeats!=None:
            self.x_con_dis_hat = self.p_x_z_dist(self.zc_infer_sample, self.zt_infer_sample, self.zy_infer_sample, self.ze_infer_sample)
            self.x_bin_dis_hat = self.p_xbin_z_dist(self.zc_infer_sample, self.zt_infer_sample, self.zy_infer_sample, self.ze_infer_sample)
        else:
            self.x_dis_hat = self.p_x_z_dist(self.zc_infer_sample, self.zt_infer_sample, self.zy_infer_sample, self.ze_infer_sample)
        
        # p(t|zt,zc)
        self.t_dis_hat = self.p_t_zc_zt_dist(self.zc_infer_sample, self.zt_infer_sample)  ## 100*1

        # p(y|t,zc,zy)
        self.y_dis_hat = self.p_y_t_zc_zy_dist(self.t_train, self.zc_infer_sample, self.zy_infer_sample)


    def fit(self,args,x_train,y_train,t_train):
        optimizer = optim.Adam(self.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # x_train, y_train, t_train = x_train.cuda(), y_train.cuda(), t_train.cuda()
        n_epoch, n_iter_per_epoch, idx = args.epochs, 10 * int(x_train.shape[0] / args.batch), list(
            range(x_train.shape[0]))  ###epoch=200
        dataset = TensorDataset(x_train,t_train, y_train)
        # dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, generator=torch.Generator(device='cuda'))

        Epoch_objective = []

        for epoch in range(n_epoch):
            np.random.shuffle(idx)
            for j in range(n_iter_per_epoch):
                # select random batch
                batch = np.random.choice(idx, args.batch)
                # x_tr, y_tr, t_tr = torch.cuda.FloatTensor(x_train[batch]), torch.cuda.FloatTensor(y_train[batch]), torch.cuda.FloatTensor(t_train[batch])
                x_tr, y_tr, t_tr = x_train[batch],y_train[batch], t_train[batch]
                x_tr, y_tr, t_tr = x_tr.cuda(), y_tr.cuda(), t_tr.cuda()
                self.forward(x_tr, t_tr, y_tr)
                objective, loss_dict = self.loss_fc()
                optimizer.zero_grad()
                # Calculate gradients
                objective.backward()
                # Update step
                optimizer.step()
                Epoch_objective.append(objective.cpu().detach().item())
        return Epoch_objective

    def loss_fc(self):
        loss = defaultdict(list)

        # p(x|zc,zt,zy,ze)
        if self.binfeats!=None:
            l_con = self.x_con_dis_hat.log_prob(self.x_train[:,self.contfeats]).sum(1)
            l_bin = self.x_bin_dis_hat.log_prob(self.x_train[:,self.binfeats]).sum(1)
            l1 = l_con + l_bin
        else:
            l1 = self.x_dis_hat.log_prob(self.x_train).sum(1)
        loss['Reconstr_x'].append(l1.cpu().detach().sum().float())

        # AUXILIARY LOSS
        # p(t|zt,zc)
        l2 = self.t_dis_hat.log_prob(self.t_train).squeeze()
        loss['Auxiliary_t'].append(l2.sum().cpu().detach().float())

        # p(y|t,zc,zy)
        l3 = self.y_dis_hat.log_prob(self.y_train).squeeze()
        loss['Auxiliary_y'].append(l3.sum().cpu().detach().float())

        #  REGULARIZATION LOSS
        # p(z) - q(z|x,t,y)
        l4 = (self.p_zc_dist.log_prob(self.zc_infer_sample) - self.zc_infer.log_prob(self.zc_infer_sample)).sum(1)
        l5 = (self.p_zt_dist.log_prob(self.zt_infer_sample) - self.zt_infer.log_prob(self.zt_infer_sample)).sum(1)
        l6 = (self.p_zy_dist.log_prob(self.zy_infer_sample) - self.zy_infer.log_prob(self.zy_infer_sample)).sum(1)
        l7 = (self.p_ze_dist.log_prob(self.ze_infer_sample) - self.ze_infer.log_prob(self.ze_infer_sample)).sum(1)
        loss['Regularization_c'].append(l4.sum().cpu().detach().float())
        loss['Regularization_t'].append(l5.sum().cpu().detach().float())
        loss['Regularization_y'].append(l6.sum().cpu().detach().float())
        loss['Regularization_e'].append(l7.sum().cpu().detach().float())

        # KL-divergence
        KLD_zc = (-torch.log(self.zc_infer.stddev) + 1 / 2 * (self.zc_infer.variance + self.zc_infer.mean ** 2 - 1)).sum(1)
        KLD_zt = (-torch.log(self.zt_infer.stddev) + 1 / 2 * (self.zt_infer.variance + self.zt_infer.mean ** 2 - 1)).sum(1)
        KLD_zy = (-torch.log(self.zy_infer.stddev) + 1 / 2 * (self.zy_infer.variance + self.zy_infer.mean ** 2 - 1)).sum(1)
        KLD_ze = (-torch.log(self.ze_infer.stddev) + 1 / 2 * (self.ze_infer.variance + self.ze_infer.mean ** 2 - 1)).sum(1)
        l8 = KLD_zc + KLD_zt + KLD_zy + KLD_ze
        loss['KL_divergence'].append(l8.sum().cpu().detach().float())

        # Total objective
        alpha1=self.args.alpha1    ## Reconstr_x
        alpha2=self.args.alpha2    ## Auxiliary_t
        alpha3=self.args.alpha3    ## Auxiliary_y
        alpha4=self.args.alpha4    ## Regularization_c
        alpha5=self.args.alpha5    ## Regularization_t
        alpha6=self.args.alpha6    ## Regularization_y
        alpha7=self.args.alpha7    ## Regularization_e
        alpha8=self.args.alpha8     ## KL_divergence
        loss_mean = torch.mean(alpha1*l1 + alpha2*l2 + alpha3*l3 + alpha4*l4 + alpha5*l5 + alpha6*l6 + alpha7*l7 - alpha8*l8)
        loss['Total'].append(loss_mean.cpu().detach().numpy())

        objective = -loss_mean

        return objective,loss

    @torch.no_grad()  ### ADRF
    def ADRF(self, x_test, t_test):
        t_grid_hat = []
        self.encode(x_test)
        for t in t_test:
            do_t = t.unsqueeze(0).expand(x_test.shape[0])
            y_out_list = []
            for _ in range(20):
                zc_infer_sample = self.zc_infer.sample()
                zy_infer_sample = self.zy_infer.sample()
                # p(y|t,zc,zy)
                y_dis_hat = self.p_y_t_zc_zy_dist(do_t, zc_infer_sample, zy_infer_sample)

                y_out = y_dis_hat.mean                   ## \mu(t,x)
                y_out = y_out.mean(0)                    ## \psi(t)  for individual
                y_out_list.append(y_out)
            y_out_mean = torch.stack(y_out_list).mean()  ##  \psi(t)
            t_grid_hat.append(y_out_mean)
        return t_grid_hat

    @torch.no_grad()
    def x_repeat_diff_Dosage(self, x_repeat, t_dosage):
        '''
        Intervention t mise: The input model x_repeat is 65 copies of the same individual, and 65 different values are medded in the same individual
        Input: x_repeat is a tensor of 65* the number of features, and a tensor of t_dosage of (65,) represents 65 different dosage values
        Output:y is (65,1) tensor
        '''
        self.encode(x_repeat)
        y_out_list = []
        for _ in range(20):  ## l = 20
            zc_infer_sample = self.zc_infer.sample()
            zy_infer_sample = self.zy_infer.sample()
            # p(y|t,zc,zy)
            y_dis_hat = self.p_y_t_zc_zy_dist(t_dosage, zc_infer_sample, zy_infer_sample)
            y_out = y_dis_hat.mean
            y_out_list.append(y_out)
        concat_tensor = torch.stack(y_out_list, dim=1)
        y_out_mean = torch.mean(concat_tensor, dim=1)

        return y_out_mean

    def plot_dis(self,dataset_name,x_test):
        self.encode(x_test)
        zc_infer_sample = self.zc_infer.sample()
        zy_infer_sample = self.zy_infer.sample()
        zt_infer_sample = self.zt_infer.sample()
        ze_infer_sample = self.ze_infer.sample()

        n = x_test.shape[0]
        x1 = np.random.normal(size=(n))
        mu = np.mean(x1)
        sigma = np.std(x1)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        y = 1 / (np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        plt.plot(x, y, 'r-', linewidth=2, label="$\mathcal{N}(0,1)$")

        mu_c = np.mean(zc_infer_sample.numpy())
        sigma_c = np.std(zc_infer_sample.numpy())
        x_c = np.linspace(mu_c - 3 * sigma_c, mu_c + 3 * sigma_c, 100)
        y_c = 1 / (np.sqrt(2 * np.pi * sigma_c ** 2)) * np.exp(-(x_c - mu_c) ** 2 / (2 * sigma_c ** 2))

        mu_t = np.mean(zt_infer_sample.numpy())
        sigma_t = np.std(zt_infer_sample.numpy())
        x_t = np.linspace(mu_t - 3 * sigma_t, mu_t + 3 * sigma_t, 100)
        y_t = 1 / (np.sqrt(2 * np.pi * sigma_t ** 2)) * np.exp(-(x_t - mu_t) ** 2 / (2 * sigma_t ** 2))

        mu_y = np.mean(zy_infer_sample.numpy())
        sigma_y = np.std(zy_infer_sample.numpy())
        x_y = np.linspace(mu_y - 3 * sigma_y, mu_y + 3 * sigma_y, 100)
        y_y = 1 / (np.sqrt(2 * np.pi * sigma_y ** 2)) * np.exp(-(x_y - mu_y) ** 2 / (2 * sigma_y ** 2))

        mu_e = np.mean(ze_infer_sample.numpy())
        sigma_e = np.std(ze_infer_sample.numpy())
        x_e = np.linspace(mu_e - 3 * sigma_e, mu_e + 3 * sigma_e, 100)
        y_e = 1 / (np.sqrt(2 * np.pi * sigma_e ** 2)) * np.exp(-(x_e - mu_e) ** 2 / (2 * sigma_e ** 2))

        plt.plot(x_c, y_c, 'c-', linewidth=2,label="$q_{\Delta}(\Delta|\\textbf{\\textit{x}})$")
        plt.plot(x_t, y_t, 'b-', linewidth=2,label="$q_{\Gamma}(\Gamma|\\textbf{\\textit{x}})$")
        plt.plot(x_y, y_y, 'g-', linewidth=2,label="$q_{\\Upsilon}(\\Upsilon|\\textbf{\\textit{x}})$")
        plt.plot(x_e, y_e, 'y-', linewidth=2,label="$q_{\\rm E}(\\rm E|\\textbf{\\textit{x}})$")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('$z_{i}$',fontsize=16)
        plt.ylabel('Density',fontsize=16)
        plt.legend(fontsize=9,frameon=False)
        ax = plt.gca()
        # ax.spines['right'].set_visible(False)
        ax.spines['right'].set_linewidth(1.2)
        # ax.spines['top'].set_visible(False)
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        plt.tight_layout()
        plt.savefig(f"{dataset_name}_Gaussian.pdf", dpi=1200)
        plt.savefig(f"{dataset_name}_Gaussian.jpeg", dpi=1200)
        plt.show()




    
