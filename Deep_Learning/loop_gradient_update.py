import torch

w = torch.tensor(10.0, requires_grad=True)
lr = 0.01

for i in range(1, 101):
    loss = w ** 2 + 20

    if w.grad is not None:
        w.grad.zero_()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    if i % 10 == 0:
        print(f"epoch={i}, w={w.item():.6f}, loss={loss.item():.6f}, grad={w.grad.item():.6f}")

print("Final result:")
print(f"w = {w.item():.6f}")
print(f"loss = {(w ** 2 + 20).item():.6f}")