import numpy as np
import matplotlib.pyplot as plt

import Utils.Analysis_Utils as AU


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
        self.lossProbe = Probe()
        self.r1r2AugSimProbe = Probe()
        self.r1AugSimProbe = Probe()
        self.r1r2AugConcProbe = Probe()
        self.r1AugConcProbe = Probe()
        self.r1VarProbe = Probe()
        self.r1CorrStrProbe = Probe()
        self.r1r2InfoBoundProbe = Probe()
        self.r1EigProbe = Probe()
        self.r1EigERankProbe = Probe()
        self.p1EntropyProbe = Probe()
        self.mz2EntropyProbe = Probe()
        self.mz2p1KLDivProbe = Probe()
        self.p1EigProbe = Probe()
        self.p1EigERankProbe = Probe()
        self.z1EigProbe = Probe()
        self.z1EigERankProbe = Probe()
        self.mz2EigProbe = Probe()
        self.mz2EigERankProbe = Probe()
        self.p1z1EigAlignProbe = Probe()
        self.p1mz2EigAlignProbe = Probe()

    # Assumes model is in eval mode, all inputs/outputs are detached, and all model/inputs/outputs are on same device
    def update_probes(self, epoch, loss, p1, z1, r1, r2, mz2):
        loss = loss.cpu().numpy()
        p1 = p1.cpu().numpy()
        z1 = z1.cpu().numpy()
        r1 = r1.cpu().numpy()
        r2 = r2.cpu().numpy()
        mz2 = mz2.cpu().numpy()

        self.epochProbe.store(epoch)

        # Probe loss throughout training
        self.lossProbe.store(float(loss))

        # Get cosine similarity between encodings
        self.r1r2AugSimProbe.store(np.mean(AU.cos_sim_bt_vecs(r1, r2)))
        self.r1AugSimProbe.store(np.mean(AU.cos_sim_within_array(r1)))
        # Get concentration (like inverse uniformity) between encodings
        # Equation derived from Uniformity measure in section 4.1.2 (page 5) of Wang/Isola for t=2: https://arxiv.org/abs/2005.10242
        self.r1r2AugConcProbe.store(np.mean(np.exp(4 * AU.cos_sim_bt_vecs(r1, r2) - 4)))
        self.r1AugConcProbe.store(np.mean(np.exp(4 * AU.cos_sim_within_array(r1) - 4)))
        # Representation variance (complete collapse measure)
        self.r1VarProbe.store(np.mean(np.var(r1, axis=0)))
        # Representation correlation strength (off-diag corr values)
        r1Corr = AU.cross_corr(r1, r1)
        self.r1CorrStrProbe.store(np.mean(np.abs(r1Corr[np.triu_indices(m=r1Corr.shape[0], k=1, n=r1Corr.shape[1])])))
        # Mutual information between views using InfoNCE bound
        self.r1r2InfoBoundProbe.store(AU.infonce_bound(r1, r2))
        # Representation encoding correlation ERank
        r1Eigvals, _ = AU.array_eigdecomp(r1, covOrCor='cor')
        # self.r1EigProbe.store(r1Eigvals)
        self.r1EigProbe.storeList = [r1Eigvals]  # Overwrite rather than append (for memory)
        self.r1EigERankProbe.store(np.exp(AU.entropy(r1Eigvals / np.sum(r1Eigvals))))

        # Get entropy and KL div between encodings
        #self.p1EntropyProbe.store(np.mean(AU.cross_ent_bt_vecs(p1, p1)))
        #self.mz2EntropyProbe.store(np.mean(AU.cross_ent_bt_vecs(mz2, mz2)))
        #self.mz2p1KLDivProbe.store(np.mean(AU.cross_ent_bt_vecs(mz2, p1 / mz2)))
        # NOTE: I'm overwriting these with None for now, normalized inputs always give the same values, so no point
        self.p1EntropyProbe.store(None)
        self.mz2EntropyProbe.store(None)
        self.mz2p1KLDivProbe.store(None)

        # Probe encoding correlation stats
        p1Eigvals, _ = AU.array_eigdecomp(p1, covOrCor='cor')
        #self.p1EigProbe.store(p1Eigvals)
        self.p1EigProbe.storeList = [p1Eigvals] # Overwrite rather than append (for memory)
        self.p1EigERankProbe.store(np.exp(AU.entropy(p1Eigvals / np.sum(p1Eigvals))))
        z1Eigvals, z1Eigvecs = AU.array_eigdecomp(z1, covOrCor='cor')
        #self.z1EigProbe.store(z1Eigvals)
        self.z1EigProbe.storeList = [z1Eigvals]  # Overwrite rather than append (for memory)
        self.z1EigERankProbe.store(np.exp(AU.entropy(z1Eigvals / np.sum(z1Eigvals))))
        mz2Eigvals, mz2Eigvecs = AU.array_eigdecomp(mz2, covOrCor='cor')
        #self.mz2EigProbe.store(mz2Eigvals)
        self.mz2EigProbe.storeList = [mz2Eigvals]  # Overwrite rather than append (for memory)
        self.mz2EigERankProbe.store(np.exp(AU.entropy(mz2Eigvals / np.sum(mz2Eigvals))))

        # Probe encoding correlation alignment
        # This method of alignment was used in Zhuo 2023 paper on Rank Differential Mechanism
        p1Corr = AU.cross_corr(p1, p1)
        z1EigvecTrans = np.tensordot(p1Corr, z1Eigvecs, axes=(-1, 0))
        mz2EigvecTrans = np.tensordot(p1Corr, mz2Eigvecs, axes=(-1, 0))
        self.p1z1EigAlignProbe.store(np.real(np.mean(AU.cos_sim_bt_vecs(z1EigvecTrans[:, :512].T, z1Eigvecs[:, :512].T))))
        self.p1mz2EigAlignProbe.store(np.real(np.mean(AU.cos_sim_bt_vecs(mz2EigvecTrans[:, :512].T, mz2Eigvecs[:, :512].T))))

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

        plt.plot(xVals, self.r1VarProbe.storeList)
        plt.xlabel('Epoch')
        plt.ylabel('Representation Average Variance')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.r1CorrStrProbe.storeList)
        plt.xlabel('Epoch')
        plt.ylabel('Representation Avg Correlation Strength')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.r1EigERankProbe.storeList)
        plt.xlabel('Epoch')
        plt.ylabel('Representation Eigval Effective Rank')
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

        #p1EigvalsArr = np.zeros((len(self.p1EigProbe.storeList[0]), len(xVals)))
        #for i in range(len(xVals)):
        #    p1EigvalsArr[:, i] = self.p1EigProbe.storeList[i]
        #for j in range(p1EigvalsArr.shape[0]):
        #    plt.plot(xVals, p1EigvalsArr[j, :], label=str(j))
        #plt.xlabel('Epoch')
        #plt.ylabel('p1 Corr Eigvals')
        #plt.grid(visible=True, which='major', axis='x')
        #plt.show()

        #z1EigvalsArr = np.zeros((len(self.z1EigProbe.storeList[0]), len(xVals)))
        #for i in range(len(xVals)):
        #    z1EigvalsArr[:, i] = self.z1EigProbe.storeList[i]
        #for j in range(z1EigvalsArr.shape[0]):
        #    plt.plot(xVals, z1EigvalsArr[j, :], label=str(j))
        #plt.xlabel('Epoch')
        #plt.ylabel('z1 Corr Eigvals')
        #plt.grid(visible=True, which='major', axis='x')
        #plt.show()

        #mz2EigvalsArr = np.zeros((len(self.mz2EigProbe.storeList[0]), len(xVals)))
        #for i in range(len(xVals)):
        #    mz2EigvalsArr[:, i] = self.mz2EigProbe.storeList[i]
        #for j in range(mz2EigvalsArr.shape[0]):
        #    plt.plot(xVals, mz2EigvalsArr[j, :], label=str(j))
        #plt.xlabel('Epoch')
        #plt.ylabel('mz2 Corr Eigvals')
        #plt.grid(visible=True, which='major', axis='x')
        #plt.show()

        plt.plot(xVals, self.p1EigERankProbe.storeList, label='p1')
        plt.plot(xVals, self.z1EigERankProbe.storeList, label='z1')
        plt.plot(xVals, self.mz2EigERankProbe.storeList, label='mz2')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Eigval Effective Rank')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.p1z1EigAlignProbe.storeList, label='p1z1')
        plt.plot(xVals, self.p1mz2EigAlignProbe.storeList, label='p1mz2')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Correlation Eigvec Alignment')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()


class Finetune_Probes:

    def __init__(self):

        self.ptEpProbe = Probe()
        self.repEigProbe = Probe()
        self.repEigERankProbe = Probe()
        self.repLolipProbe = Probe()
        self.knnAccProbe = Probe()
        self.clnTrainAccProbe = Probe()
        self.clnTestAccProbe = Probe()
        self.advAccProbe = Probe()
        #self.atkVecEntProbe = Probe()

    def update_probes(self, ptEp, repBank, avgRepLolip, knnAcc, clnTrainAcc, clnTestAcc, advAcc):
        repBank = repBank.cpu().numpy()

        self.ptEpProbe.store(ptEp)

        repEigvals, _ = AU.array_eigdecomp(repBank, covOrCor='cor')
        self.repEigProbe.storeList = [repEigvals]
        self.repEigERankProbe.store(np.exp(AU.entropy(repEigvals / np.sum(repEigvals))))
        self.repLolipProbe.store(avgRepLolip)

        self.knnAccProbe.store(knnAcc)
        self.clnTrainAccProbe.store(clnTrainAcc)
        self.clnTestAccProbe.store(clnTestAcc)
        self.advAccProbe.store(advAcc)

        # Measure cossim of every atk vector to a random vector, count vectors with the same cossim, calculate entropy
        # Note perturbTens is np.float32, so it's necessary to convert the rand vector to np.float32
        # If rand vector is np.float64, tiny differences (1e-9) appear and screw up np.unique()
        #perturbTens = perturbTens.cpu().numpy()
        #simVals = AU.cos_sim_bt_arrays(np.random.rand(1, perturbTens.shape[1]).astype(np.float32), perturbTens)
        #_, counts = np.unique(simVals, return_counts=True)
        #entropy = 0
        #for count in counts:
        #    entropy -= count / len(simVals) * np.log(count / len(simVals))
        #self.atkVecEntProbe.store(entropy)

    def plot_probes(self):

        xVals = self.ptEpProbe.storeList

        plt.plot(xVals, self.clnTrainAccProbe.storeList)
        plt.xlabel('Pretrain Epoch')
        plt.ylabel('Clean Cls Acc')
        plt.grid(visible=True, which='major', axis='x')
        plt.show()

        plt.plot(xVals, self.clnTestAccProbe.storeList)
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

        #plt.plot(xVals, self.atkVecEntProbe.storeList)
        #plt.xlabel('Pretrain Epoch')
        #plt.ylabel('Atk Vector Alignment Entropy')
        #plt.grid(visible=True, which='major', axis='x')
        #plt.show()

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
