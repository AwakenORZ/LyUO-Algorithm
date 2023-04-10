import numpy as np
import scipy.io as sio  # import scipy.io for .mat file I/

from memory import MemoryDNN

import math
import torch

V = 20
ksi = 10 ** (-26)
zeta = 10 ** (-27)
Alpha = 0.1  # address 1 Mbit tasks profit
Beta = 0.6  # MEC power price 1J
N = 10  # number of users
n = 10000  # number of time frames
phi = 100  # number of cpu cycles for processing 1 bit data
psi = 100  # number of MEC cpu cycles for processing 1 bit data
F_max = 4*10 ** 3  # MEC's maximum frequency MHz
W = 2  # bandwidth MHz
DTaskL = np.zeros((n, N))  # local achieved task amount
DTaskO = np.zeros((n, N))  # remote offloading task amount
DTaskM = np.zeros(n)  # MEC achieved task amount
DTaskT = np.zeros((n, N))  # achieved task amount
Energyo = np.zeros((n, N))
Energyl = np.zeros((n, N))
Energy = np.zeros((n, N))  # energy consumption


def Action_Quantization(m, P_off, e):
    # return M offloading actions
    m = m[:] / np.sum(m)
    temp_e = e[:] - ksi * f_loc[:] ** 3 * 10 ** 18
    temp_e[temp_e[:] < 0] = 0

    klist = []
    klist.append(m)
    sparetime = np.zeros(N)
    sparetime[:] = m[:] - temp_e[:] / P_off[:]
    sparetime[sparetime[:] < 0] = 0
    add4 = np.zeros(N)
    add5 = np.zeros(N)

    m[:] = m[:] - sparetime[:]
    add1 = np.sum(sparetime)
    entotal = np.sum(temp_e)
    if add1 == 0:
        return klist

    sparetime[sparetime[:] > 0] = 1
    # add2 = np.sum(sparetime)
    # k = N - add2 + 1

    for i in range(N):
        add3 = np.zeros(N)
        if sparetime[i] == 0:
            add3[i] = add1
            klist.append(m + add3)
            add4[i] = add1 * (temp_e[i] / entotal)
            add5[i] = add1 / (N - np.sum(sparetime))
    klist.append(m + add4)
    klist.append(m + add5)
    return klist


def plot_rate(rate_his, rolling_intv=50, ylabel='Normalized Computation Rate'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)  # rate_his transformed to array type
    df = pd.DataFrame(rate_his)

    mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15, 8))

    plt.plot(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    plt.fill_between(np.arange(len(rate_array)) + 1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values),
                     np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color='b', alpha=0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.show()


# generate racial fading channel with power h and Line of sight ratio factor
def racial_mec(h, factor):
    n = len(h)
    beta = np.sqrt(h * factor)  # LOSE channel amplitude
    sigma = np.sqrt(h * (1 - factor) / 2)  # scattering sdv
    x = np.multiply(sigma * np.ones(n), np.random.randn(n)) + beta * np.ones(n)  # 矩阵对应元素相乘
    y = np.multiply(sigma * np.ones(n), np.random.randn(n))
    g = np.power(x, 2) + np.power(y, 2)  # np.power(x, y)为计算x的y次方 包括但不限于数、数组、矩阵
    return g


def Alg(strategy, h, A, Q, e, Qmec, f_loc, P_off, i):
    ch_fact = 10 ** 10  # scaling factors to avoid numerical precision problems
    d_fact = 10 ** 6
    N0 = W * d_fact * (10 ** (-17.4)) * (10 ** (-3)) * ch_fact  # noise power in watt
    R = W * np.log2(1 + h[:] / N0 * P_off[:])

    cur_e = e[:] + 1
    cur_e = cur_e - 1
    cur_tasko = np.zeros(N)
    cur_taskl = np.zeros(N)
    cur_taskt = np.zeros(N)
    cur_energy = np.zeros(N)
    cur_energyo = np.zeros(N)
    cur_energyl = np.zeros(N)

    for j in range(N):
        T_loc = 1
        if Q[j] + A[j] - V * Alpha < 0:
            T_loc = 0
            cur_taskl[j] = 0
            cur_energyl[j] = 0
        else:
            # local computing
            if cur_e[j] < ksi * f_loc[j] ** 3 * 10 ** 18:
                T_loc = cur_e[j] / (ksi * f_loc[j] ** 3 * 10 ** 18)
            cur_taskl[j] = f_loc[j] * T_loc / phi
            if cur_taskl[j] > Q[j]:
                cur_taskl[j] = Q[j]
                T_loc = cur_taskl[j] * phi / f_loc[j]
            cur_energyl[j] = ksi * f_loc[j] ** 3 * 10 ** 18 * T_loc
            cur_e[j] = cur_e[j] - cur_energyl[j]

        if Q[j] + A[j] - Qmec < 0:
            cur_tasko[j] = 0
            cur_energyo[j] = 0
        else:
            # remoted offloading
            cur_tasko[j] = R[j] * strategy[j]
            if Q[j] - cur_taskl[j] < cur_tasko[j]:
                cur_tasko[j] = Q[j] - cur_taskl[j]
            cur_energyo[j] = P_off[j] * cur_tasko[j] / R[j]
            if cur_e[j] < cur_energyo[j]:
                cur_energyo[j] = cur_e[j]
                cur_tasko[j] = cur_energyo[j] * R[j] / P_off[j]

        cur_taskt[j] = cur_tasko[j] + cur_taskl[j]
        cur_energy[j] = cur_energyo[j] + cur_energyl[j]

    # solve Fmec_optimal
    A1 = np.minimum(F_max, Qmec * psi)
    A2 = (np.sqrt(
        1 / (psi ** 4) + 12 * (Qmec + np.sum(cur_tasko[:]) + V * Alpha) / psi * V * Beta * zeta * 10 ** 18) - 1 / (
                      psi ** 2)) / (6 * V * Beta * zeta * 10 ** 18)
    # A2 = np.sqrt((Qmec+V*Alpha)/(3*V*Beta*zeta*10**18))
    F_val = np.minimum(A1, A2)

    utility = np.sum((Q[:] + A[:] - Alpha * V) * cur_taskl[:]) + np.sum(
        (Q[:] + A[:] - Qmec) * cur_tasko[:]) + V * Beta * zeta * 10 ** 18 * F_val ** 3 + (
                      Qmec + np.sum(cur_tasko[:]) - V * Alpha) * F_val / psi - F_val ** 2 / (2 * psi ** 2)

    F_val = np.around(F_val, decimals=6)
    utility = np.around(utility, decimals=6)

    return utility, F_val, cur_tasko, cur_taskl, cur_energy, cur_energyo, cur_energyl


if __name__ == '__main__':
    K = 1  # initialize K = 1
    Memory = 1024  # capacity of memory structure
    Delta = 32  # update interval for adaptive K
    CHFACT = 10 ** 10  # the factor for scaling channel value 10^10
    nu = 1000  # energy queue factor
    energy_threshold = np.ones(N) * 50  # energy consumption threshold in J per time slot
    arrival_lambda = 2.4 * np.ones(N)  # average data arrival, 3Mbps per user0-7
    arrival_energy = 1 * np.ones(N) * 0.08  # average energy arrival
    w = [1.5 if i % 2 == 0 else 1 for i in range(N)]  # weights for each user

    print('#user = %d, #channel=%d, K=%d, Memory = %d, Delta = %d' % (N, n, K, Memory, Delta))

    # initialize data
    channel = np.zeros((n, N))  # chanel gains
    TaskA = np.zeros((n, N))  # arrival data size
    EnergyA = np.zeros((n, N))  # energy harvesting size

    # generate channel h0
    dist_v = np.linspace(start=120, stop=255, num=N)
    Ad = 3
    fc = 915 * 10 ** 6
    loss_exponent = 3  # path loss exponent
    light = 3 * 10 ** 8
    h0 = np.ones(N)
    for j in range(0, N):
        h0[j] = Ad * (light / 4 / math.pi / fc / dist_v[j]) ** loss_exponent

    model = MemoryDNN(net=[N * 3, 256, 128, N], learning_rate=0.0001, training_interval=5, batch_size=128,
                      memory_size=Memory)

    model_strategy = []  # store the offloading strategy
    model_put = []  # store the NN put (new)
    k_idx_opt = []  # store the index of optimal offloading actor
    Q = np.zeros((n, N))  # device data queue in Mbits
    Qmec = np.zeros(n)  # MEC data queue in Mbits
    f_loc = np.zeros(N)  # Local frequency
    F_MEC = np.zeros(n)  # MEC frequency
    P_off = np.zeros(N)  # offloading power
    e = np.zeros((n, N))  # battery energy queue in mJ
    Obj = np.zeros(n)  # utility value
    Yield = np.zeros(n)

    f_loc[:] = 150  # np.random.rand(N) * 100 + 100  # local computing frequency 100~200MHz 150

    for i in range(n):
        # Program running progress
        if i % (n // 10) == 0:
            print("%0.1f" % (i / n))

        i_index = i
        P_off[:] = np.random.rand(N) * 0.2 + 0.3  # maximum transmit power 300~500mW
        # real-time channel generation
        h_tmp = racial_mec(h0, 0.3)
        # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        h = h_tmp * CHFACT
        channel[i, :] = h
        # real-time arrival generation
        TaskA[i, :] = np.random.exponential(arrival_lambda)
        EnergyA[i, :] = np.random.exponential(arrival_energy)

        if i_index > 0:
            # update queues
            Q[i_index, :] = Q[i_index - 1, :] + TaskA[i_index - 1, :] - DTaskT[i_index - 1, :]
            e[i_index, :] = e[i_index - 1, :] + EnergyA[i_index - 1, :] - Energy[i_index - 1, :]
            Qmec[i_index] = Qmec[i_index - 1] + np.sum(DTaskO[i_index - 1, :]) - DTaskM[i_index - 1]

            # assert Q and Qmec are positive due to float error
            Q[i_index, Q[i_index, :] < 0] = 0
            e[i_index, e[i_index, :] < 0] = 0
            e[i_index, e[i_index, :] > 50] = 50
            Qmec[Qmec[i_index] < 0] = 0

        if i_index == 2000:
            print("")

        NN_input = np.concatenate((h, Q[i_index, :] / 100, EnergyA[i_index, :] / 100))
        # 1) 'Actor module'
        # generate a batch of actions
        # m_list = Action_Quantization(model.decode(NN_input), P_off, e[i_index, :])
        m_list = Action_Quantization(model.decode(NN_input), P_off, EnergyA[i_index, :])

        r_list = []  # all results of candidate offloading modes
        v_list = []  # the objective values of candidate offloading modes

        for m in m_list:
            # 2) 'Critic module'
            # allocate resource for all generated offloading strategies saved in m_list
            # r_list.append(Alg(m, h, TaskA[i_index, :], Q[i_index, :], e[i_index, :], Qmec[i_index], f_loc, P_off, i_index))
            r_list.append(
                Alg(m, h, TaskA[i_index, :], Q[i_index, :], EnergyA[i_index, :], Qmec[i_index], f_loc, P_off, i_index))
            v_list.append(r_list[-1][0])

        # record the index of the largest reward
        k_idx_opt.append(np.argmax(v_list))

        # 3) 'Policy update module'
        # encode the mode with the largest reward
        model.encode(NN_input, m_list[k_idx_opt[-1]])
        model_strategy.append(m_list[k_idx_opt[-1]])

        # store max result
        # utility, F_val, cur_tasko, cur_taskl, cur_energy, cur_energyo, cur_energyl
        Obj[i_index], F_MEC[i_index], DTaskO[i_index], DTaskL[i_index], Energy[i_index], Energyo[i_index], Energyl[
            i_index] = r_list[k_idx_opt[-1]]
        DTaskM[i_index] = F_MEC[i_index] / psi
        DTaskT[i_index, :] = DTaskL[i_index, :] + DTaskO[i_index, :]
        Yield[i_index] = Alpha * (DTaskM[i_index] + np.sum(DTaskL[i_index])) - Beta * zeta * (
                F_MEC[i_index] * 10 ** 6) ** 3
    model.plot_cost()

    cons = []
    cons.append(np.average(Q))
    cons.append(np.average(DTaskT))
    cons.append(np.average(Qmec))
    cons.append(np.average(Yield))
    print(cons)

    plot_rate(Q.sum(axis=1) / N, 100, 'Average Data Queue')
    plot_rate(Energy.sum(axis=1) / N, 100, 'Average Energy Consumption')
    plot_rate(EnergyA.sum(axis=1) / N, 100, 'Average EnergyA')
    plot_rate(DTaskT.sum(axis=1) / N, 100, 'Average Battery Energy queue')
    plot_rate(Qmec, 100, 'MEC Data queue')
    plot_rate(Yield, 100, 'System Yield')
    print(np.sum(EnergyA[:, :]))
    print(np.sum(Energy[:, :]))
    # print(np.sum(e[:, :]))

    # sio.savemat('./result_%d.mat' % N,
    #            {'input_h': channel / CHFACT, 'data_arrival': TaskA, 'data_queue': Q, 'mec_queue': Qmec,
    #             'local_task': DTaskL, 'offloading_task': DTaskO, 'address_task': DTaskT, 'MEC_task': DTaskM,
    #             'local_energy': Energyl, 'offloading_energy': Energyo, 'F_MEC': F_MEC,
    #             'offloading_strategy': model_strategy, 'yield': Yield})

    # sio.savemat('./resultV_%d.mat' % V, {'input_h': channel / CHFACT, 'data_arrival': TaskA, 'energy_arrival': EnergyA,
    #                                    'data_queue': Q, 'mec_queue': Qmec, 'F_MEC': F_MEC, 'energy': Energy,
    #                                    'offloading_strategy': model_strategy, 'address_task': DTaskT, 'yield': Yield})
    # print("")
