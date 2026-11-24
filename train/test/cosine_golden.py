"""The PyTorch peer of the cosine tests of `loss_test.agc`: `torch.nn.functional.cosine_similarity` of two fixed
`[3, 4]` tensors, the MSE of it against fixed scores, and the gradients of that loss, in `float64`.

Manual, not run by the tests: it needs `torch` (2.6.0+cpu was used). Its printed lists are the constants of the test.

    python -I test/cosine_golden.py
"""

import torch

A = [[0.3, -1.2, 2.5, 0.7], [-0.4, 1.1, 0.2, -0.9], [1.5, 0.1, -0.6, 0.8]]
B = [[1.0, 0.5, -0.3, 0.2], [-0.2, 0.9, 0.4, -1.3], [-0.7, 0.6, 1.2, 0.1]]
SCORES = [0.2, 0.9, 0.4]


def main():
    a = torch.tensor(A, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(B, dtype=torch.float64, requires_grad=True)
    cos = torch.nn.functional.cosine_similarity(a, b)
    loss = torch.nn.functional.mse_loss(cos, torch.tensor(SCORES, dtype=torch.float64))
    loss.backward()
    print("torch", torch.__version__)
    print("cos", cos.tolist())
    print("loss", loss.item())
    print("grad_a", a.grad.flatten().tolist())
    print("grad_b", b.grad.flatten().tolist())


main()
