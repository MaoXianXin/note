用Cross-Entropy Loss来解决multi-class classification problem

反向传播之前记得清楚已经存在的梯度:
net.zero_grad()
loss.backward()
Before we feed the input to our network model, we need to clear the previous gradient

torch.from_numpy()   numpy转化为tensor