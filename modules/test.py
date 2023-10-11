from AED import AttentionEncoderDecoder
import torch


aed = AttentionEncoderDecoder()
print(aed)

input = torch.rand(2,500,80)
x_lengths = torch.tensor([198,500])
label = torch.rand(2, 30,5000)
label_lengths = torch.tensor([14,30])
out = aed(input, x_lengths, label, label_lengths)
print(out,out.size())
