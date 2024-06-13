import numpy as np
from scipy.integrate import romb
import torch
import pandas as pd

def get_device():
    # return "cuda:0"
    return "cpu"

def pre_paraIHDP():
    path = 'dataset/ihdp/ihdp.csv'
    ihdp = pd.read_csv(path)
    ihdp = ihdp.to_numpy()
    ihdp = ihdp[:, 2:27]  # delete the first column (data idx)/ delete the second coloum (treatment)
    ihdp = torch.from_numpy(ihdp)
    ihdp = ihdp.float()

    n_feature = ihdp.shape[1]   ## 25
    n_data = ihdp.shape[0]      ## 747
    for _ in range(n_feature):
        minval = min(ihdp[:, _]) * 1.
        maxval = max(ihdp[:, _]) * 1.
        ihdp[:, _] = (1. * (ihdp[:, _] - minval))/maxval

    # cate_idx = torch.tensor([3,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    cate_idx1 = torch.tensor([3,6,7,8,9,10,11,12,13,14])      ## dis1
    cate_idx2 = torch.tensor([15,16,17,18,19,20,21,22,23,24]) ## dis2
    cate_mean1 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    cate_mean2 = torch.mean(ihdp[:, cate_idx1], dim=1).mean()
    return cate_mean1

def get_IHDPpatient_outcome(t, x, cate_mean1):
    alpha = 5.
    cate_idx1 = torch.tensor([3, 6, 7, 8, 9, 10, 11, 12, 13, 14])  ## dis1
    # only x1, x3, x4 are useful
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[4]
    x5 = x[5]

    # v1
    factor1 = 0.5
    factor2 = 1.5

    # v2
    factor1 = 1.5
    factor2 = 0.5

    # original
    # factor1 = 1.
    # factor2 = 1.

    y = 1. / (1.2 - t) * torch.sin(t * 3. * 3.14159) * (
                factor1 * torch.tanh((torch.sum(x[cate_idx1]) / 10. - cate_mean1) * alpha) +
                factor2 * torch.exp(0.2 * (x1 - x5)) / (0.1 + min(x2, x3, x4)))
    return y + torch.randn(1)[0] * 0.5

def get_NEWSpatient_outcome(t, x):
    t = t.numpy()
    x = x.numpy()
    num_feature = x.shape[0]
    np.random.seed(5)
    v1 = np.random.randn(num_feature)
    v1 = v1/np.sqrt(np.sum(v1**2))
    v2 = np.random.randn(num_feature)
    v2 = v2/np.sqrt(np.sum(v2**2))
    v3 = np.random.randn(num_feature)
    v3 = v3/np.sqrt(np.sum(v3**2))
    res1 = max(-2, min(2, np.exp(0.3 * (np.sum(3.14159 * np.sum(v2 * x) / np.sum(v3 * x)) - 1))))
    res2 = 20. * (np.sum(v1 * x))
    res = 2 * (4 * (t - 0.5) ** 2 * np.sin(0.5 * 3.14159 * t)) * (res1 + res2)
    # y = torch.from_numpy(res) + torch.randn(1)[0] * np.sqrt(0.5)
    y = torch.tensor(res) + torch.randn(1)[0] * np.sqrt(0.5)
    return y

def get_SIMU1patient_outcome(t, x):
    # only x1, x3, x4 are useful
    x1 = x[0]
    x3 = x[2]
    x4 = x[3]
    x6 = x[5]
    y = torch.cos((t-0.5) * 3.14159 * 2.) * (t**2 + (4.*max(x1, x6)**3)/(1. + 2.*x3**2)*torch.sin(x4))
    y += torch.randn(1)[0] * 0.5
    return y


def get_model_predictions(num_treatments, test_data, model):  ### GIKS
    x = test_data['x']
    t = test_data['t']
    d = test_data['d']
    I_logits = model.forward(dosage=d, t=t, x=x)[1]
    return I_logits.cpu().detach()

def get_DRVAE_predictions(num_treatments, test_data, model):  ###DRVAE
    x = test_data['x']
    t = test_data['t']
    d = test_data['d']
    y_out_mean = model.x_repeat_diff_Dosage(x, d)
    return y_out_mean.cpu().detach()


def compute_ihdpORnews_eval_metrics(dataset_name, test_patients, num_treatments, model, train=False):
    mises = []
    ites = []
    dosage_policy_errors = []
    policy_errors = []
    pred_best = []

    samples_power_of_two = 6
    num_integration_samples = 2 ** samples_power_of_two + 1
    step_size = 1. / num_integration_samples
    treatment_strengths = np.linspace(np.finfo(float).eps, 1, num_integration_samples)
    treatment_strengths = torch.FloatTensor(treatment_strengths).to(get_device(), dtype=torch.float64)
    if "ihdp" in dataset_name:
        cate_mean1 =  pre_paraIHDP()

    for patient in test_patients:
        if train and len(pred_best) > 10:
            return np.sqrt(np.mean(mises)), np.sqrt(np.mean(dosage_policy_errors)), np.sqrt(np.mean(policy_errors)), np.mean(ites)

        assert num_treatments == 1, "We deal wil univariate treatment only"

        for treatment_idx in range(num_treatments):   ## num_treatments =1
            test_data = dict()
            test_data['x'] = torch.repeat_interleave(patient.view(1, -1), num_integration_samples, dim=0)
            test_data['t'] = torch.ones(num_integration_samples).to(get_device(), dtype=torch.float64) * treatment_idx
            test_data['d'] = treatment_strengths

            # pred_dose_response = get_model_predictions(num_treatments=num_treatments, test_data=test_data, model=model)
            pred_dose_response = get_DRVAE_predictions(num_treatments=num_treatments, test_data=test_data, model=model)

            if "ihdp" in dataset_name:
                true_outcomes = [get_IHDPpatient_outcome(d, patient, cate_mean1) for d in treatment_strengths]
            if "news" in dataset_name:
                true_outcomes = [get_NEWSpatient_outcome(d, patient) for d in treatment_strengths]
            if "simu" in dataset_name:
                true_outcomes = [get_SIMU1patient_outcome(d, patient) for d in treatment_strengths]

            true_outcomes = torch.FloatTensor(true_outcomes)   ##（65,）


            mise = romb(torch.square(true_outcomes.squeeze() - pred_dose_response.squeeze()), dx=step_size)
            inter_r = true_outcomes.squeeze() - pred_dose_response.squeeze()
            ite = torch.mean(inter_r ** 2)
            mises.append(mise)
            ites.append(ite)

            max_dosage_pred, max_dosage = treatment_strengths[torch.argmax(pred_dose_response)],  treatment_strengths[torch.argmax(true_outcomes)]
            max_y_pred, max_y = true_outcomes[torch.argmax(pred_dose_response)], torch.max(true_outcomes)

            dosage_policy_error = (max_y - max_y_pred) ** 2
            dosage_policy_errors.append(dosage_policy_error.item())

    # For 1 treatment case, both dpe and pe should be the same
    return np.mean(np.sqrt(mises)), np.mean(np.sqrt(dosage_policy_errors)), np.mean(ites)