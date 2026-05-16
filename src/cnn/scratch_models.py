class ScratchSequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        out = x

        for layer in self.layers:
            out = layer.forward(out)

        return out