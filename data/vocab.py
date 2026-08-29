class Solution:
    def build_vocab(self, text):
        chars = sorted(set(text))

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        return stoi, itos

    def encode(self, text, stoi):
        return [stoi[ch] for ch in text]

    def decode(self, ids, itos):
        return ''.join(itos[i] for i in ids)