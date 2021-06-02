import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 字体管理器
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
if __name__ == "__main__":
    pro = np.loadtxt("../w1w2pro.txt")
    ls = np.loadtxt("../w1w2ls.txt")

    pro_s = []
    ls_s = []
    nor_s = []
    # sample data
    for i in range(len(pro)):
        if i % 10 == 0:
            pro_s.append(pro[i])
            ls_s.append(ls[i])

    x_data = [i for i in range(len(pro_s))]
    nor_s = [1 for i in range(len(pro_s))]
    # print(pro)
    # print(ls)

    ln1, = plt.plot(x_data, pro_s,linestyle='dotted')
    ln2, = plt.plot(x_data, ls_s,linestyle='dashed')
    ln3, = plt.plot(x_data, nor_s,color="blue")

    # my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")

    # plt.title("Pro与Ls对比")  # 设置标题及字体

    plt.legend(handles=[ln1, ln2,ln3], labels=['pro', 'ls','nor'])

    ax = plt.gca()
    ax.spines['right'].set_color('none')  # right边框属性设置为none 不显示
    ax.spines['top'].set_color('none')  # top边框属性设置为none 不显示
    # ax.set_xlabel("样本编号")
    # ax.set_ylabel("|w1-w2|")
    plt.show()
