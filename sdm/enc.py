from sdm.utils import random_sdr, sdr_distance


class TokenEncoder:
    """
    Encodes tokens to sparse distributed representations and maintains the mapping.
    """

    def __init__(self, n, p):
        """
        Initialize a TokenEncoder.

        Args:
            n (int): Dimension of the sparse vectors
            p (int): Number of active bits in the sparse vectors
        """
        self.n = n
        self.p = p
        self.tok2sdr = {}
        self.sdr2tok = {}

    def encode(self, token):
        """
        Encode a token to a sparse distributed representation.

        Args:
            token (str): Toekn to encode

        Returns:
            numpy.ndarray: Sparse distributed representation
        """
        if token in self.tok2sdr:
            return self.tok2sdr[token]

        # Generate a new random SDR
        sdr = random_sdr(self.n, self.p)

        # Store the mappings
        self.tok2sdr[token] = sdr
        sdr_key = tuple(sdr)
        self.sdr2tok[sdr_key] = token

        return sdr

    def decode(self, sdr):
        """
        Decode a sparse distributed representation to a token.
        Returns the closest matching token if exact match isn't found.

        Args:
            sdr (numpy.ndarray): Sparse distributed representation

        Returns:
            str: Decoded token or <|unk|> if not found
        """
        sdr_key = tuple(sdr)

        # Exact match
        if sdr_key in self.sdr2tok:
            return self.sdr2tok[sdr_key]

        # the harsh variant would return the sdr or <|unk|> right away
        # return self.sdr2tok.get(sdr_key, "<|unk|>")

        # No exact match, find the closest one
        best_distance = float("inf")
        best_token = "<|unk|>"

        for token, known_sdr in self.tok2sdr.items():
            distance = sdr_distance(sdr, known_sdr)
            if distance < best_distance:
                best_distance = distance
                best_token = token

                # If very close, return immediately
                if best_distance < 0.1:  # TODO: pick a reasonable value
                    return best_token

        return best_token
