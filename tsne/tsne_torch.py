import numpy as np
import matplotlib.pyplot as pyplot
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--xfile", type=str, default="features.txt", help="file name of feature stored")
parser.add_argument("--yfile", type=str, default="labels.txt", help="file name of label stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")

opt = parser.parse_args()
print("get choice from args", opt)
xfile = opt.xfile
yfile = opt.yfile

if opt.cuda:
    print("set use cuda")
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.DoubleTensor)

cls_idx = {
    "bao_lu_la_ji": 0,
    "cheng_san_jing_ying": 1,
    "gu_ding_tan_fan": 2,
    "ling_san_la_ji": 3,
    "kua_men_ying_ye": 4,
    "liu_dong_tan_fan": 5,
    "luan_dui_wu_liao": 6,
    "luan_la_tiao_fu": 7,
    "luan_shai_yi_wu": 8,
    "luan_she_guang_gao_pai": 9
}
cls_color = {
    0: "#FFFF00",  # 黄色
    1: "#FF0000",  # 大红
    2: "#FF00FF",  # 粉色
    3: "#B9D3EE",  # 浅蓝色
    4: "#C7C7C7",  # 灰色
    5: "#B3EE3A",  # 青色
    6: "#242424",  # 黑色
    7: "#00FFFF",  # 亮蓝色
    8: "#00CD00",  # 绿色
    9: "#9400D3"  # 深紫色
}
cls_color = {
    0: "red",  # 占道堆物
    1: "blue", # 暴露垃圾
    2: "green",# 乱设广告怕
    3: "black" # 违规摆摊
}
cls_idx = {
    0: 1,
    1: 3,
    2: 3,
    3: 1,
    4: 3,
    5: 3,
    6: 0,
    7: 2,
    8: 2,
    9: 2
}

def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X * X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i] + n_list[i + 1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i] + n_list[i + 1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=50):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i + 1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should not have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    print(X.shape)
    X = pca_torch(X, initial_dims)
    (n, d) = X.shape
    max_iter = 500
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.  # early exaggeration
    print("get P shape", P.shape)
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y * Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")

    X = np.loadtxt(xfile)
    X = torch.Tensor(X)
    labels = np.loadtxt(yfile).tolist()
    # print(labels)
    labels = [cls_color[cls_idx[i]] for i in labels]
    # print(labels)
    # labels = [cls_color[i] for i in labels]
    # print(labels)
    # confirm that x file get same number point than label file
    # otherwise may cause error in scatter
    print(len(X), len(labels))
    assert (len(X[:, 0]) == len(X[:, 1]))
    assert (len(X) == len(labels))

    with torch.no_grad():
        Y = tsne(X, 2, 50, 20.0)

    if opt.cuda:
        Y = Y.cpu().numpy()

    pyplot.title("resnet101 avg ")
    pyplot.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pyplot.show()
