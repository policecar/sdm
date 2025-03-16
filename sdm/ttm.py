import numpy as np

from sdm.utils import empty_sdr, random_sdr, sdr_overlap, sdr_union
from sdm.sdr import TriadicMemory


class TriadicTemporalMemory:
    """
    Temporal Memory implementation using three Triadic Memories
    Following Peter Overmann's Mathematica implementation

    The algorithm creates a temporal memory circuit using three triadic memories:
    M0: encodes bigrams (current input with previous input)
    M1: encodes context (previous bigram with previous context)
    M2: stores predictions (x and y to predict next input)
    """

    def __init__(self, n, p):
        """
        Initialize Temporal Memory with three Triadic Memory instances

        Args:
            n (int): Dimension of the sparse vectors
            p (int): Number of active bits in the sparse vectors
        """
        # Initialize the three triadic memory instances as in the Mathematica code
        self.M0 = TriadicMemory(n, p)  # Encodes bigrams
        self.M1 = TriadicMemory(n, p)  # Encodes context
        self.M2 = TriadicMemory(n, p)  # Stores predictions

        self.n = n
        self.p = p

        # Initialize circuit state variables with null/empty vectors
        self.reset()

    def reset(self):
        """Reset all state variables to empty SDRs"""
        null_vector = empty_sdr()
        self.i = null_vector  # Current input
        self.j = null_vector  # Previous input
        self.y = null_vector  # Bigram
        self.c = null_vector  # Context
        self.u = null_vector  # Bundle of previous input and context
        self.v = null_vector  # Bigram (for prediction)
        self.prediction = null_vector  # Predicted next token

    def process(self, inp):
        """
        Process a new input through the temporal memory circuit

        Args:
            inp (numpy.ndarray): Input SDR

        Returns:
            numpy.ndarray: Prediction for the next input
        """
        # Save previous input and update current
        self.j = self.i
        self.i = inp

        # Get or create a bigram representation
        bigram = self.M0.query_Z(self.i, self.j)

        # If the bigram doesn't sufficiently overlap with previous input,
        # create a new random bigram
        if (
            len(self.j) > 0
            and sdr_overlap(self.M0.query_Y(self.i, bigram), self.j) < self.p
        ):
            bigram = random_sdr(self.n, self.p)
            self.M0.store(self.i, self.j, bigram)

        # Bundle previous input with previous context
        x = sdr_union(self.y, self.c)
        # np.sort(np.union1d(a, b)).astype(np.uint32)

        # Update bigram
        self.y = bigram

        # Store new prediction if necessary
        if not np.array_equal(self.prediction, self.i):
            self.M2.store(self.u, self.v, self.i)

        # Create or retrieve context
        self.c = self.M1.query_Z(x, self.y)

        # If context doesn't sufficiently overlap with x, create new random context
        if sdr_overlap(self.M1.query_X(self.y, self.c), x) < self.p:
            self.c = random_sdr(self.n, self.p)
            self.M1.store(x, self.y, self.c)

        # Update prediction state variables
        self.u = x
        self.v = self.y

        # Make prediction
        self.prediction = self.M2.query_Z(self.u, self.v)

        return self.prediction
