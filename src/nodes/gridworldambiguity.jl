using RxInfer
using TensorOperations

export GridWorldAmbiguity

struct GridWorldAmbiguity end

@node GridWorldAmbiguity Stochastic [A, orientation, selfloc, doorloc, keyloc, doorstate, keystate]
