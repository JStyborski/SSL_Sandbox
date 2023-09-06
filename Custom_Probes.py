import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import Model_Analysis.Analysis_Utils as Analysis_Utils
import Adversarial.Local_Lip as Local_Lip


class Probe:
    def __init__(self):
        self.storeList = []
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def store(self, x):
        self.storeList.append(x)

    def runSum(self, x):
        self.sum += x

    def runAvg(self, x):
        self.sum += x
        self.count += 1
        self.avg = self.sum / self.count


class Pretrain_Probes:

    def __init__(self):

        self.epochProbe = Probe()

        # Probe loss throughout training
        self.lossProbe = Probe()

        # Get cosine similarity between encodings
        self.r1r2AugSimProbe = Probe()
        self.r1AugSimProbe = Probe()

        # Get entropy and KL div between encodings
        self.p1EntropyProbe = Probe()
        self.mz2EntropyProbe = Probe()
        self.mz2p1KLDivProbe = Probe()

        # Probe encoding correlation stats
        self.p1EigProbe = Probe()
        self.p1EigERankProbe = Probe()
        self.p1EigAUCProbe = Probe()
        self.z1EigProbe = Probe()
        self.z1EigERankProbe = Probe()
        self.z1EigAUCProbe = Probe()
        #self.r1EigProbe = Probe()
        #self.r1EigERankProbe = Probe()
        #self.r1EigAUCProbe = Probe()
        self.mz2EigProbe = Probe()
        self.mz2EigERankProbe = Probe()
        self.mz2EigAUCProbe = Probe()

        # Probe encoding correlation alignment
        self.p1z1EigAlignProbe = Probe()
        self.p1mz2EigAlignProbe = Probe()

        # Get smoothness stats
        self.r1LolipProbe = Probe()
        #self.z1LolipProbe = Probe()
        #self.p1LolipProbe = Probe()

    # Assumes model is in eval mode, all inputs/outputs are detached, and all model/inputs/outputs are on same device
    def update_probes(self, epoch, model, loss, p1, z1, r1, r2, mz2, x1):
        loss = loss.cpu().numpy()
        p1 = p1.cpu().numpy()
        z1 = z1.cpu().numpy()
        r1 = r1.cpu().numpy()
        r2 = r2.cpu().numpy()
        mz2 = mz2.cpu().numpy()

        self.epochProbe.store(epoch)

        self.lossProbe.store(float(loss))

        self.r1r2AugSimProbe.store(np.mean(Analysis_Utils.cos_sim_bt_vecs(r1, r2)))
        self.r1AugSimProbe.store(np.mean(Analysis_Utils.cos_sim_within_array(r1)))

        self.p1EntropyProbe.store(np.mean(Analysis_Utils.cross_ent_bt_vecs(p1, p1)))
        self.mz2EntropyProbe.store(np.mean(Analysis_Utils.cross_ent_bt_vecs(mz2, mz2)))
        self.mz2p1KLDivProbe.store(np.mean(Analysis_Utils.cross_ent_bt_vecs(mz2, p1 / mz2)))

        p1Eigvals, _ = Analysis_Utils.array_eigdecomp(p1, covOrCor='cor')
        self.p1EigProbe.store(p1Eigvals)
        self.p1EigERankProbe.store(np.exp(Analysis_Utils.entropy(p1Eigvals / np.sum(p1Eigvals))))
        self.p1EigAUCProbe.store(Analysis_Utils.area_under_curve(p1Eigvals, np.ones(len(p1Eigvals)) / len(p1Eigvals)))
        z1Eigvals, z1Eigvecs = Analysis_Utils.array_eigdecomp(z1, covOrCor='cor')
        self.z1EigProbe.store(z1Eigvals)
        self.z1EigERankProbe.store(np.exp(Analysis_Utils.entropy(z1Eigvals / np.sum(z1Eigvals))))
        self.z1EigAUCProbe.store(Analysis_Utils.area_under_curve(z1Eigvals, np.ones(len(z1Eigvals)) / len(z1Eigvals)))
        # r1Eigvals, r1Eigvecs = Analysis_Utils.array_cov_eig(r1, covOrCor='cor')
        # self.r1EigProbe.store(r1Eigvals)
        # self.r1EigERankProbe.store(np.exp(Analysis_Utils.entropy(r1Eigvals / np.sum(r1Eigvals))))
        # self.r1EigAUCProbe.store(Analysis_Utils.area_under_curve(r1Eigvals, np.ones(len(r1Eigvals)) / len(r1Eigvals)))
        mz2Eigvals, mz2Eigvecs = Analysis_Utils.array_eigdecomp(mz2, covOrCor='cor')
        self.mz2EigProbe.store(mz2Eigvals)
        self.mz2EigERankProbe.store(np.exp(Analysis_Utils.entropy(mz2Eigvals / np.sum(mz2Eigvals))))
        self.mz2EigAUCProbe.store(Analysis_Utils.area_under_curve(mz2Eigvals, np.ones(len(mz2Eigvals)) / len(mz2Eigvals)))

        # This method of alignment was used in Zhuo 2023 paper on Rank Differential Mechanism
        p1Corr = Analysis_Utils.cross_corr(p1, p1)
        z1EigvecTrans = np.tensordot(p1Corr, z1Eigvecs, axes=(-1, 0))
        mz2EigvecTrans = np.tensordot(p1Corr, mz2Eigvecs, axes=(-1, 0))
        self.p1z1EigAlignProbe.store(np.mean(Analysis_Utils.cos_sim_bt_vecs(z1EigvecTrans[:, :512].T, z1Eigvecs[:, :512].T)))
        self.p1mz2EigAlignProbe.store(np.mean(Analysis_Utils.cos_sim_bt_vecs(mz2EigvecTrans[:, :512].T, mz2Eigvecs[:, :512].T)))

        randIdxArr = np.random.choice(range(x1.size(0)), size=8, replace=False).tolist()
        randInpsTens = x1[randIdxArr, :]
        #avgP1Lolip, _ = Local_Lip.maximize_local_lip(model, randInpsTens, 0.003, 0.01, 1, np.inf, 8, 1, 10, outIdx=0)
        # self.p1LolipProbe.store(avgP1Lolip)
        #avgZ1Lolip, _ = Local_Lip.maximize_local_lip(model, randInpsTens, 0.003, 0.01, 1, np.inf, 8, 1, 10, outIdx=1)
        # self.z1LolipProbe.store(avgZ1Lolip)
        avgR1Lolip, _ = Local_Lip.maximize_local_lip(model, randInpsTens, 0.003, 0.01, 1, np.inf, 8, 1, 10, outIdx=2)
        self.r1LolipProbe.store(avgR1Lolip)

    def plot_probes(self):

        xVals = range(len(self.lossProbe.storeList))

        plt.plot(xVals, self.lossProbe.storeList)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.r1r2AugSimProbe.storeList, label='r1r2-SameSrc')
        plt.plot(xVals, self.r1AugSimProbe.storeList, label='r1-DiffSrc')
        plt.xlabel('Epoch')
        plt.ylabel('Average Cosine Similarity')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.p1EntropyProbe.storeList, label='p1')
        plt.plot(xVals, self.mz2EntropyProbe.storeList, label='mz2')
        plt.xlabel('Epoch')
        plt.ylabel('Encoding Entropy')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.mz2p1KLDivProbe.storeList, label='mz2p1')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Encoding KL Divergence')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        p1EigvalsArr = np.zeros((len(self.p1EigProbe.storeList[0]), len(xVals)))
        for i in range(len(xVals)):
            p1EigvalsArr[:, i] = self.p1EigProbe.storeList[i]
        for j in range(p1EigvalsArr.shape[0]):
            plt.plot(xVals, p1EigvalsArr[j, :], label=str(j))
        plt.xlabel('Epoch')
        plt.ylabel('p1 Corr Eigvals')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        z1EigvalsArr = np.zeros((len(self.z1EigProbe.storeList[0]), len(xVals)))
        for i in range(len(xVals)):
            z1EigvalsArr[:, i] = self.z1EigProbe.storeList[i]
        for j in range(z1EigvalsArr.shape[0]):
            plt.plot(xVals, z1EigvalsArr[j, :], label=str(j))
        plt.xlabel('Epoch')
        plt.ylabel('z1 Corr Eigvals')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        # r1EigvalsArr = np.zeros((len(self.r1EigProbe.storeList[0]), len(xVals)))
        # for i in range(len(xVals)):
        #    r1EigvalsArr[:, i] = self.r1EigProbe.storeList[i]
        # for j in range(r1EigvalsArr.shape[0]):
        #    plt.plot(xVals, r1EigvalsArr[j, :], label=str(j))
        # plt.xlabel('Epoch')
        # plt.ylabel('r1 Corr Eigvals')
        # plt.grid(visible=True, which='major', axis='x')
        # plt.show()

        mz2EigvalsArr = np.zeros((len(self.mz2EigProbe.storeList[0]), len(xVals)))
        for i in range(len(xVals)):
            mz2EigvalsArr[:, i] = self.mz2EigProbe.storeList[i]
        for j in range(mz2EigvalsArr.shape[0]):
            plt.plot(xVals, mz2EigvalsArr[j, :], label=str(j))
        plt.xlabel('Epoch')
        plt.ylabel('mz2 Corr Eigvals')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.p1EigERankProbe.storeList, label='p1')
        plt.plot(xVals, self.z1EigERankProbe.storeList, label='z1')
        #plt.plot(xVals, self.r1EigERankProbe.storeList, label='r1')
        plt.plot(xVals, self.mz2EigERankProbe.storeList, label='mz2')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Eigval Effective Rank')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.p1EigAUCProbe.storeList, label='p1')
        plt.plot(xVals, self.z1EigAUCProbe.storeList, label='z1')
        #plt.plot(xVals, self.r1EigAUCProbe.storeList, label='r1')
        plt.plot(xVals, self.mz2EigAUCProbe.storeList, label='mz2')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Eigval AUC')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.p1z1EigAlignProbe.storeList, label='p1z1')
        plt.plot(xVals, self.p1mz2EigAlignProbe.storeList, label='p1mz2')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Eigvec Alignment')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        #plt.plot(xVals, self.p1LolipProbe.storeList, label='p1')
        #plt.plot(xVals, self.z1LolipProbe.storeList, label='z1')
        plt.plot(xVals, self.r1LolipProbe.storeList, label='r1')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Avg Local Lip')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()


class Finetune_Probes:

    def __init__(self):

        self.ptEpProbe = Probe()
        self.clnAccProbe = Probe()
        self.knnAccProbe = Probe()
        self.advAccProbe = Probe()
        self.atkVecEntProbe = Probe()

    def update_probes(self, ptEp, clnAcc, knnAcc, advAcc, perturbTens):

        perturbTens = perturbTens.cpu().numpy()

        self.ptEpProbe.store(ptEp)
        self.clnAccProbe.store(clnAcc)
        self.knnAccProbe.store(knnAcc)
        self.advAccProbe.store(advAcc)

        # Measure cossim of every atk vector to a random vector, count vectors with the same cossim, calculate entropy
        # Note perturbTens is np.float32, so it's necessary to convert the rand vector to np.float32
        # If rand vector is np.float64, tiny differences (1e-9) appear and screw up np.unique()
        simVals = Analysis_Utils.cos_sim_bt_arrays(np.random.rand(1, perturbTens.shape[1]).astype(np.float32), perturbTens)
        _, counts = np.unique(simVals, return_counts=True)
        entropy = 0
        for count in counts:
            entropy -= count / len(simVals) * np.log(count / len(simVals))
        self.atkVecEntProbe.store(entropy)

    def plot_probes(self):

        xVals = self.ptEpProbe.storeList

        plt.plot(xVals, self.clnAccProbe.storeList)
        plt.xlabel('Pretrain Epoch')
        plt.ylabel('Clean Cls Acc')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.knnAccProbe.storeList)
        plt.xlabel('Pretrain Epoch')
        plt.ylabel('KNN Acc')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.advAccProbe.storeList)
        plt.xlabel('Pretrain Epoch')
        plt.ylabel('Adversarial Cls Acc')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.atkVecEntProbe.storeList)
        plt.xlabel('Pretrain Epoch')
        plt.ylabel('Atk Vector Alignment Entropy')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        #fig, ax = plt.subplots(1, 1)
        #def animate_zVals(i):
        #    ax.scatter(self.zValsProbe.storeList[i][:, 0], self.zValsProbe.storeList[i][:, 1])
        #    ax.set_title('Pretrain Epoch {}'.format(xVals[i]))
        #    ax.set_xlabel('Encoding Dim 1')
        #    ax.set_ylabel('Encoding Dim 2')

        #ani = FuncAnimation(fig, animate_zVals, frames=len(xVals), repeat=False)
        #ani.save('zVals.gif')

        #ani = FuncAnimation(fig, animate_zEigvals, frames=len(xVals), repeat=False)
        #ani.save('zEigvals.gif')
