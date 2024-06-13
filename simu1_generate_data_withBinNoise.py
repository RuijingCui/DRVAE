import os
import numpy as np

from data.simu1withNoise import simu_data1withBINNoise
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generate simulate data')
    # parser.add_argument('--save_dir', type=str, default='dataset/simu1_withNoise_5con_5bin', help='dir to save generated data')
    parser.add_argument('--num_eval', type=int, default=100, help='num of dataset for evaluating the methods')
    parser.add_argument('--num_tune', type=int, default=20, help='num of dataset for tuning the parameters')

    parser.add_argument('--num_CONnoise', type=int, default=5, help='num of continuous noise adding to the data')
    parser.add_argument('--loc_noises', type=int, default=2, help='mean of continuous noise adding to the data')
    parser.add_argument('--scale_noises', type=int, default=10, help='std of continuous noise adding to the data')

    parser.add_argument('--num_BINnoise', type=int, default=10, help='num of binary noise adding to the data')

    args = parser.parse_args()
    # save_path = args.save_dir   ### 'dataset/simu1'
    save_path = f'dataset/simu1_Noise_{args.num_CONnoise}con_{args.loc_noises}loc_{args.scale_noises}std_{args.num_BINnoise}bin'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for _ in range(args.num_tune):   ## 20
        print('generating tuning set: ', _)
        data_path = os.path.join(save_path, 'tune', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data1withBINNoise(500, 200, args.num_CONnoise,args.loc_noises,args.scale_noises,args.num_BINnoise)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())

    for _ in range(args.num_eval):  ## 100
        print('generating eval set: ', _)
        data_path = os.path.join(save_path, 'eval', str(_))
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_matrix, test_matrix, t_grid = simu_data1withBINNoise(500, 200, args.num_CONnoise,args.loc_noises,args.scale_noises,args.num_BINnoise)

        data_file = os.path.join(data_path, 'train.txt')
        np.savetxt(data_file, train_matrix.numpy())
        data_file = os.path.join(data_path, 'test.txt')
        np.savetxt(data_file, test_matrix.numpy())
        data_file = os.path.join(data_path, 't_grid.txt')
        np.savetxt(data_file, t_grid.numpy())


