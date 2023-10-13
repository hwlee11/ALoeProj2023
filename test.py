from modules.AED import AttentionEncoderDecoder
import torch
import torch.nn.functional as F

'''
class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError

class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()
'''

aed = AttentionEncoderDecoder()
#decoding = GreedyDecoder(1,1999)

input = torch.rand(2,500,80)
x_lengths = torch.tensor([198,500])
label = torch.rand(2, 30).to(torch.int64)
label_lengths = torch.tensor([14,30])
out,out_mask = aed(input, x_lengths, label, label_lengths)
#print(out_mask, out_mask.size())

h, h_mask = aed.encoder_forward(input,x_lengths)
tokens = torch.tensor([[0],[0]])
tokens_lengths = torch.tensor([1,1])
############# greedy decoding
eos = 0
sum_logprobs = torch.zeros(2,2000)
for i in range(10):
    out, out_mask = aed.decode_one_step(tokens, tokens_lengths, h, h_mask)
    next_token = out.argmax(dim=1)
    logprobs = F.log_softmax(out[:,:,-1:],dim=1).squeeze(-1)
    print("logp",logprobs,logprobs.size())
    sum_logprobs += logprobs.squeeze(0)
    print(sum_logprobs)

    next_token[tokens[:, -1] == eos] = eos
    tokens = torch.cat([tokens,next_token],dim=-1)
    endOfDecode = (tokens[:,-1] == eos).all()
    print(tokens)
    print(endOfDecode)
    if endOfDecode:
        break
