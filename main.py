# -*- coding: utf-8 -*-
"""
Created on Fri May 15 10:43:37 2020
Title: 
@author: Dr. Tian Guo
"""

import os
import shutil

import torch
import numpy as np

from utils.util import plot_training_rst, plot_loss_rst

from utils.load_data import build_dataset
from utils.train_func import train_func

if __name__ == "__main__":

    save_path = "/home/zq/program/ORR_prediction-master/save_results/"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.mkdir(save_path)

    # ========== setting & loading ==========
    flag = "E_O"
    X_data_train, y_data_train, X_data_test, y_data_test, X_data_predict = build_dataset(flag)

    # ========== training & select best performing model ==========
    loop_num = 50
    error_flag_test = 1
    for _ in range(loop_num):

        loss_rec_train, error_rec_train, loss_rec_test, error_rec_test, model = train_func(flag)

        if error_rec_test[-1] <= error_flag_test:
            error_flag_train = error_rec_train[-1]  # update the error flag
            error_flag_test = error_rec_test[-1]  # update the error flag
            train_error = error_rec_train  # update the training error
            test_error = error_rec_test  # update the testing error
            model_sav = model

            # ========== prediction & final test ==========
    train_eval = []  # training evaluation

    model_sav.eval()
    with torch.no_grad():
        for coord in X_data_train:
            outputs = model_sav(coord)
            train_eval.append(outputs)

    test_eval = []  # testing evaluation
    model_sav.eval()
    with torch.no_grad():
        for coord in X_data_test:
            outputs = model_sav(coord)
            test_eval.append(outputs)

    predicts = []  # prediction
    model_sav.eval()


    with torch.no_grad():
        for coord in X_data_predict:
            outputs = model_sav(coord)
            predicts.append(outputs)

    # ========== print & save results ==========
    plot_training_rst(train_error, test_error)
    plot_loss_rst(loss_rec_train, loss_rec_test)
    # predicts.
    # print(predicts)

    # 保存 error

    np.savetxt(save_path + "ML_results_" + flag + "_train_error.txt", train_error)
    np.savetxt(save_path + "ML_results_" + flag + "_test_error.txt", test_error)



    predicts_np_list = [i.numpy() for i in predicts]
    np.savetxt(save_path + "ML_results_" + flag + "_preditions.txt", predicts_np_list)

    train_eval_np_list = [i.numpy() for i in train_eval]
    np.savetxt(save_path + "ML_results_" + flag + "_train.txt", train_eval_np_list)

    test_eval_np_list = [i.numpy() for i in test_eval]
    np.savetxt(save_path + "ML_results_" + flag + "_test.txt", test_eval_np_list)

    torch.save(model_sav.state_dict(), save_path + "atomic_net_" + flag + ".pkl")
