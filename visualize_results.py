from matplotlib import pyplot
palette = pyplot.get_cmap('Set1')
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 32,
}
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.unicode_minus'] = False
# mpl.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rc('font',family='Times New Roman')
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def draw_line(AMSE_dict, metrics):

    i = 0
    # colors = ["#0071c2", "#d76515","#EDB11A","#7E318A","#78AB31","#2A77AC","#D55535","#E9262E"]
    # colors = ["#FABB6E", "#d76515","#FC8002","#7E318A","#369F2D","#2A77AC","#0071c2","#E9262E"]
    # colors = ["#16499D", "#EFE342","#7D4195","#EF7D1A","#369F2D","#2A77AC","#2CB8BB","#E71F19"]
    # colors = ["#1A2F54", "#FFD700", "#800080", "#FFA500", "#008000", "#ADD8E6", "#00CED1", "#FF0000"]
    colors = ["#4B0082", "#FFA07A", "#7FFF00", "#87CEEB", "#DAA520", "#FF007F", "#4B0082", "#00FF7F"]
    markers = ["o","^","s","p","h","v","d","*",]
    for model_name, mean_std in AMSE_dict.items():
        color = palette(i)
        avg = np.array(mean_std[0])
        std = np.array(mean_std[1])
        r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))
        r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))
        iters = list(range(len(avg)))
        if model_name=="DRVAE":
            plt.plot(iters, avg, color="#E71F19", marker = "*",markersize=10, label=model_name, linewidth=2.0)
        else:
            plt.plot(iters, avg, color=colors[i], marker = markers[i], markersize=10, label= model_name, linewidth=2.0)
        # plt.fill_between(iters, r1, r2, color=color, alpha=0.2)
        plt.xticks(iters,fontsize=22)
        # plt.yticks(np.linspace(np.min(avg), np.max(avg), 5),fontsize=22)
        if metrics == "AMSE": plt.yticks(np.linspace(0.05,0.35, 5),fontsize=22)
        if metrics == "MISE": plt.yticks(np.linspace(0.6,1.1, 5),fontsize=22)
        if metrics == "DPE": plt.yticks(np.linspace(0.85,1.2, 5),fontsize=22)
        if metrics == "ITE": plt.yticks(np.linspace(0.4,1.4, 5),fontsize=22)
        plt.gca().yaxis.set_major_formatter('{:.3f}'.format)
        i += 1

    plt.xlabel("Simu($k$)", fontdict={'family' : 'Times New Roman', 'size': 22})
    tick_positions = np.linspace(0, 4, 5)
    plt.xticks(tick_positions, [1, 2, 3, 4, 5], fontsize=22)
    if metrics == "AMSE": plt.ylabel(metrics, fontdict={'family' : 'Times New Roman', 'size': 22})
    if metrics == "MISE": plt.ylabel("$\sqrt{\\rm MISE}$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if metrics == "DPE": plt.ylabel("$\sqrt{\\rm DPE}$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if metrics == "ITE": plt.ylabel("$i$-MSE", fontdict={'family' : 'Times New Roman', 'size': 22})
    if metrics =="AMSE":
        plt.legend(loc=(4/16, 3/9),prop={'family' : 'Times New Roman', 'size': 12},frameon=False)
    ax = plt.gca()

    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig(f"figures/{metrics}.pdf", dpi=600)
    plt.savefig(f"figures/{metrics}.jpeg", dpi=1200)
    plt.show()

def read_data(sheet_name):
    AMSE = pd.read_excel("figures/results_forPlot.xlsx", sheet_name=sheet_name)
    model_names = AMSE.iloc[:, 0].tolist()    ## ['Tarnet', 'Drnet', 'Vcnet', 'Vcnet_tr', 'TransTEE', 'TransTEE_tr', 'GIKS', 'DRVAE']
    dataset_names = AMSE.columns.tolist()[1:] ## ['simu5', 'simu10', 'simu15', 'simu20', 'simu25']
    AMSE_dict = {model: ([],[]) for model in model_names}
    for index, row in AMSE.iterrows():
        row_array = row.values
        for item in row_array[1:]:
            split_data = item.split("+-")
            AMSE_dict[row_array[0]][0].append(float(split_data[0]))
            AMSE_dict[row_array[0]][1].append(float(split_data[1]))
    return AMSE_dict

def para_sense(para_name):
    if para_name == "alpha":
        para = np.array([0.018834582436829804, 0.014732886478304863, 0.01791587108746171, 0.018089074082672595, 0.02282832534983754])
    elif para_name == "beta":
        para = np.array([0.018250081967562438, 0.01846755687147379, 0.018827962502837182, 0.014732886478304863, 0.018937154859304427])
    elif para_name == "gama":
        para = np.array([0.019005256704986095, 0.018935541901737452, 0.018967803660780193, 0.014732886478304863, 0.06363875167444348])
    elif para_name == "deta":
        para = np.array([0.4020780652761459, 0.014732886478304863, 0.018201472982764245, 0.017315111868083478, 0.01941355150192976])
    elif para_name == "lamda":
        para = np.array([0.018377167452126742, 0.018564961198717356, 0.01887682070955634, 0.014732886478304863, 0.01895800232887268])
    # beta_std = np.array([0.006785383705323339, 0.00693557726816543,0.007422159592154719])

    iters = list(range(len(para)))
    plt.plot(iters, list(para), color="blue",linewidth=3.5)
    red_line_index = np.argmin(para)
    plt.axvline(x=red_line_index, color='red', linestyle='--')
    # plt.fill_between(iters, r1, r2, color=palette(1), alpha=0.2)
    tick_positions = np.linspace(0, 4, 5)
    plt.xticks(tick_positions, [0.0, 0.1, 0.5, 1.0, 10.0], fontsize=22)
    if para_name == "alpha": plt.xlabel(r"$\alpha$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if para_name == "beta": plt.xlabel(r"$\beta$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if para_name == "gama": plt.xlabel(r"$\gamma$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if para_name == "deta": plt.xlabel(r"$\delta$", fontdict={'family' : 'Times New Roman', 'size': 22})
    if para_name == "lamda": plt.xlabel(r"$\lambda$", fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.ylabel("AMSE", fontdict={'family' : 'Times New Roman', 'size': 22})
    if para_name == "deta": plt.yticks(np.linspace(np.min(para), np.max(para), 5), fontsize=22)
    if para_name != "deta": plt.yticks([0.01,0.02,0.03,0.04,0.05,0.06], fontsize=22)
    plt.gca().yaxis.set_major_formatter('{:.2f}'.format)
    ax = plt.gca()
    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig(f"figures/{para_name}.pdf", dpi=600)
    plt.savefig(f"figures/{para_name}.jpeg", dpi=1200)
    plt.show()

def simu5_DRcurve():


    file_path = 'DRcurve/DRVAE_simu1_Noise_5con_2loc_10std_10bin_0_0.005170146934688091.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y = df['y_truth']
    plt.plot(t, y,color="#171BFF",linewidth=2,label='Truth',zorder=5)


    file_path = "DRcurve/Vcnet_simu1_Noise_5con_2loc_10std_10bin_5_mse=0.01117516029626131.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#0EFF0C",linewidth=2, label='Vcnet',zorder=4)

    file_path = "DRcurve/Drnet_simu1_Noise_5con_2loc_10std_10bin_5_mse=0.05586547777056694.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Drnet')

    file_path = "DRcurve/Tarnet_simu1_Noise_5con_2loc_10std_10bin_3_mse=0.05115906894207001.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Tarnet')


    file_path = "DRcurve/GIKS_simu1_Noise_5con_2loc_10std_10bin_5_mse=0.008808902952100012.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    # plt.scatter(t, y_hat, label='GIKS')
    plt.plot(t, y_hat,color="#737373",linewidth=2,  label='GIKS')  ##

    file_path = 'DRcurve/DRVAE_simu1_Noise_5con_2loc_10std_10bin_0_0.005170146934688091.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')

    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#FF1624",linewidth=2,label='DRVAE',zorder=6)

    plt.xlabel('Treatment $t$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.ylabel('Response $\psi(t)$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.legend(fontsize=18,frameon=False)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax = plt.gca()

    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig(f"figures/plot_simu5.pdf", dpi=1200)
    plt.savefig(f"figures/plot_simu5.jpeg", dpi=1200)
    plt.show()

def ihdp_DRcurve():


    file_path = 'DRcurve/DRVAE_ihdp_9_0.042689044028520584.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y = df['y_truth']
    plt.plot(t, y,color="#171BFF",linewidth=2,label='Truth',zorder=5)

    file_path = "DRcurve/Vcnet_ihdp_7_0.37995320558547974.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#0EFF0C",linewidth=2, label='Vcnet',zorder=4)

    file_path = "DRcurve/Drnet_ihdp_4_0.9328492879867554.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Drnet')

    file_path = "DRcurve/Tarnet_ihdp_5_0.7281515598297119.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Tarnet')

    # file_path = "DRcurve/GIKS_ihdp_5_mse=0.2384432233894111.xlsx"
    file_path = "DRcurve/GIKS_ihdp_6_mse=0.077552526178615.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    # plt.scatter(t, y_hat, label='GIKS')
    plt.plot(t, y_hat,color="#737373",linewidth=2,  label='GIKS')  ##


    file_path = 'DRcurve/DRVAE_ihdp_9_0.042689044028520584.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#FF1624",linewidth=2,label='DRVAE',zorder=6)

    plt.xlabel('Treatment $t$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.ylabel('Response $\psi(t)$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.legend(fontsize=16,frameon=False)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax = plt.gca()

    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig(f"figures/plot_ihdp.pdf", dpi=1200)
    plt.savefig(f"figures/plot_ihdp.jpeg", dpi=1200)
    plt.show()

def news_DRcurve():


    file_path = 'DRcurve/DRVAE_news_7_0.001966165378689766.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y = df['y_truth']
    plt.plot(t, y,color="#171BFF",linewidth=2,label='Truth',zorder=5)

    file_path = "DRcurve/Vcnet_news_6_0.0020021137315779924.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#0EFF0C",linewidth=2, label='Vcnet',zorder=4)

    file_path = "DRcurve/Drnet_news_7_0.04903298243880272.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Drnet')

    file_path = "DRcurve/Tarnet_news_7_0.04914764687418938.xlsx"
    df = pd.read_excel(file_path)
    t = df['t']
    y_hat = df['y_hat']
    plt.scatter(t, y_hat,alpha=1, zorder=2, s=10, linewidth=2, label='Tarnet')

    # file_path = "DRcurve/GIKS_ihdp_5_mse=0.2384432233894111.xlsx"
    file_path = "DRcurve/GIKS_news_9_mse=0.06074934219014336.xlsx"
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    # plt.scatter(t, y_hat, label='GIKS')
    plt.plot(t, y_hat,color="#737373",linewidth=2,  label='GIKS')  ##


    # file_path = 'DRcurve/DRVAE_news_8_0.004269763827323914.xlsx'
    # file_path = 'DRcurve/DRVAE_news_6_0.0038690611254423857.xlsx'
    # file_path = 'DRcurve/DRVAE_news_9_0.005475710146129131.xlsx'
    file_path = 'DRcurve/DRVAE_news_7_0.001966165378689766.xlsx'
    df = pd.read_excel(file_path)
    df = df.sort_values(by='t')
    t = df['t']
    y_hat = df['y_hat']
    plt.plot(t, y_hat,color="#FF1624",linewidth=2,label='DRVAE',zorder=6)

    plt.xlabel('Treatment $t$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.ylabel('Response $\psi(t)$',fontdict={'family' : 'Times New Roman', 'size': 22})
    plt.legend(loc="upper center",fontsize=16,frameon=False)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    ax = plt.gca()

    ax.tick_params(axis='both', direction='in')
    plt.tight_layout()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.savefig(f"figures/plot_news.pdf", dpi=1200)
    plt.savefig(f"figures/plot_news.jpeg", dpi=1200)
    plt.show()


if __name__ == '__main__':

    ##
    for sheet_name in ["AMSE","MISE","DPE","ITE"]:
        AMSE_dict = read_data(sheet_name)
        draw_line(AMSE_dict, metrics=sheet_name)

    ##
    for para_name in ["alpha" ,  "beta", "gama", "deta", "lamda"]:
        para_sense(para_name)

    simu5_DRcurve()  ## simu1
    ihdp_DRcurve()   ## ihdp
    news_DRcurve()   ## news

