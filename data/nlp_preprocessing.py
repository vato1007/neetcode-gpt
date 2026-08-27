class Solution:
    def get_dataset(self, positive, negative):
        sentences = positive + negative

        words = set()
        for sentence in sentences:
            words.update(sentence.split())

        vocab = {}
        for i, word in enumerate(sorted(words), 1):
            vocab[word] = i

        max_len = max(len(sentence.split()) for sentence in sentences)

        result = []

        for sentence in sentences:
            row = [vocab[word] for word in sentence.split()]
            row += [0] * (max_len - len(row))
            result.append(row)

        return result