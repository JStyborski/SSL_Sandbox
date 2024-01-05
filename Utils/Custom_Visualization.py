import torch


# Between a baseline image and the target image, calculate the intermediate gradient values and discretely integrate
def integrated_gradients(imgTens, baseImgTens, model, targetIdx, nSteps):

    # Other inputs covered in run_baseline_integ_gradients
    # baseline is the baseline image numpy array, same size as imgArr

    # This is a list of the image arrays representing the steps along a linear path between baseline and input images
    imgDelta = imgTens - baseImgTens
    interpList = [baseImgTens + float(i) / nSteps * imgDelta for i in range(0, nSteps + 1)]
    interpTens = torch.cat(interpList, dim=0)
    interpTens.requires_grad = True

    outPred, _, _, _ = model(interpTens)
    outputVal = torch.sum(outPred[:, targetIdx])

    model.zero_grad()
    outputVal.backward()
    gradsTens = interpTens.grad.detach()

    # Explanation: We have calculated our gradients at 51 (or other) points from baseline to input. This is like f(x)
    # Now we seek to integrate f(x) from baseline to input, so we do a discrete Riemann sum
    # Since we have a constant interval (imgDelta), we can just do avg(f(x)) * (image-baseline)
    # The line below gives the avg(f(x)) using a central Riemann sum.
    # e.g., for 5 steps 0 to 5: avg(f(x)) ~ ((f(0) + f(1)) / 2 + (f(1) + f(2)) / 2 + ... + (f(4) + f(5)) / 2) / 5
    # Equivalently, as in the format below: (f(1) + f(2) + f(3) + f(4) + (f(0) + f(5)) / 2) / 5
    avgGrads = (torch.sum(gradsTens[1:-1], dim=0) + (gradsTens[0] + gradsTens[-1]) / 2) / (gradsTens.size(0) - 1)

    # Final multiplication of the central Riemann sum
    integGrad = imgDelta.detach() * avgGrads

    return integGrad


def multi_baseline_IG(imgTens, model, targetIdx, nSteps, nBaselines):

    igList = []
    if nBaselines == 'zero':
        print('  Zero Baseline')
        integGrad = integrated_gradients(imgTens, 0.0 * imgTens, model, targetIdx, nSteps)
        igList.append(integGrad)

    else:
        for i in range(nBaselines):
            print('  Random Baseline {}'.format(i + 1))
            integGrad = integrated_gradients(imgTens, torch.randn_like(imgTens), model, targetIdx, nSteps)
            igList.append(integGrad)

    avgIG = torch.mean(torch.cat(igList, dim=0), dim=0)

    return avgIG



