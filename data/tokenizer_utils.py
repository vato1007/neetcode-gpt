class Solution:
    def tokenize_numbers(self, numbers, vocab):
        result = []

        # Reverse vocabulary: id -> token
        # We only need the token strings here.
        tokens = set(vocab.keys())

        for number in numbers:
            text = str(number)
            tokenized = []
            i = 0

            while i < len(text):
                best = None

                # Find the longest vocabulary token starting at i
                for token in tokens:
                    if text.startswith(token, i):
                        if best is None or len(token) > len(best):
                            best = token

                if best is not None:
                    tokenized.append(best)
                    i += len(best)
                else:
                    # No match -> consume one character
                    tokenized.append(text[i])
                    i += 1

            result.append(tokenized)

        return result

    def count_tokens(self, text, vocab):
        tokens = set(vocab.keys())
        i = 0
        count = 0

        while i < len(text):
            best = None

            for token in tokens:
                if text.startswith(token, i):
                    if best is None or len(token) > len(best):
                        best = token

            if best is not None:
                i += len(best)
            else:
                i += 1

            count += 1

        return count

    def fertility_score(self, text, vocab):
        words = text.split()

        if not words:
            return 0.0

        token_count = self.count_tokens(text, vocab)
        return round(token_count / len(words), 4)