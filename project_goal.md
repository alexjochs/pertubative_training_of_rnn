Goal and task. Train a recurrent core on Moving MNIST next-step prediction using Evolution Strategies (ES) rather than backprop through time. Each sample is a sequence of frames x1..xT; objective is to predict x_{t+1} (or its embedding) from past frames, starting with T=20.

**Target Hardware**: This code is designed to run on a node with 8x H200 GPUs. Efficiency and distributed training stability are key priorities.


Data. Use a standard Moving MNIST generator (64×64 grayscale, two digits is common but start with one digit to reduce difficulty). Train on sequences of length 20, predict the 21st frame (later extend to multi-step).

Encoder (frozen). Build a small CNN that maps each frame (1×64×64) to a 128-d embedding z_t. Use a simple 3-layer stride-2 conv stack with ReLU and global average pooling to produce z_t ∈ R^{128}. Initialize randomly and freeze. (No classifier head; the embedding is the pooled feature vector.)

Reservoir core. Hidden size N=2000 initially. Recurrent weight matrix W0 is sparse Erdos–Renyi (or fixed in-degree) with target in-degree k=50. Nonzero weights initialized with variance scaled by 1/√k. Use a leaky tanh cell:
h_{t+1} = (1−α) h_t + α · tanh( (W0 + ΔW) h_t + Win z_t + b )
Use a single scalar leak α to start (e.g., 0.2–0.5); b can be zero initially.

Input projection. Win is dense with shape (N×128) so every hidden unit receives input signal. Initialize Win with a standard small random init; keep Win fixed initially.

Trainable part of the recurrent core (adapter). Use a low-rank recurrent adapter:
ΔW = U Vᵀ with U,V ∈ R^{N×r}, rank r=32. Initialize U and V to small values (so ΔW is near zero at start). Train only U and V with ES; keep W0 and Win fixed.

Output and loss. Predict the next-step embedding z_{t+1} from the current hidden state h_t using a linear decoder:
ẑ_{t+1} = Wout h_t + bout, with Wout ∈ R^{128×N}.
Train Wout,bout with standard supervised learning (SGD/Adam) because it is stable and cheap. The ES loop optimizes only U,V. Loss is MSE in embedding space summed over t=1..T−1 (teacher forcing with true z_t as input).

ES training loop (high-level). Let θ be the vector of all trainable ES parameters (U and V flattened). Each ES iteration:
	1.	Sample K noise vectors ε_i ~ N(0, I). Use antithetic pairs, so evaluate θ+σ ε_i and θ−σ ε_i.
	2.	For each candidate parameter set, run a forward-only evaluation on a fixed minibatch (same data for all candidates in that iteration) to compute the loss.
	3.	Convert losses to weights (either standard ES gradient estimate or rank-based weights). Form an update direction:
ĝ ≈ (1/(2Kσ)) Σ_i (L(θ+σ ε_i) − L(θ−σ ε_i)) ε_i
	4.	Update θ with an optimizer (SGD or Adam on θ) using ĝ. (This is not backprop; it is an optimizer step on the ES-estimated gradient.)
	5.	Optionally evaluate the unperturbed θ on a held-out minibatch periodically for tracking.

Multi-GPU parallelization. Distribute candidate evaluations across 8 H200s. Each GPU gets a slice of candidates (for example, if K=32 antithetic pairs, that is 64 evaluations; assign 8 evaluations per GPU). Each evaluation is independent: run the same minibatch through the frozen CNN and the reservoir with that candidate’s U,V, compute loss, return scalar loss. Aggregate losses on CPU (or one process), compute ĝ, update θ, broadcast updated θ to workers for the next iteration.

Stability and efficiency guardrails. Early terminate rollout if ||h_t|| exceeds a threshold or if NaNs appear; assign a large fixed loss. Clip losses or use rank-based weighting so rare catastrophic losses do not dominate the ES update. Keep σ modest initially so most candidates stay stable; adjust σ upward if progress stalls or downward if most candidates explode.

Schedule and scaling. Start with N=2000, r=32, k=50, T=20, embedding dim 128, one digit Moving MNIST. Once the loop is stable and learning, increase difficulty in this order: two digits, longer horizons (T=50, 100), larger N (5000), larger r (64), then optionally allow training of α (scalar) or per-neuron leak α_i in addition to U,V.

Logging and checkpoints. Track training loss on the minibatch used for ES updates, periodic evaluation loss on a fixed validation set, fraction of exploding candidates per iteration, and norms of U,V and h_t. Save θ, Wout, and RNG seeds periodically so runs are reproducible.

Deliverable for the first working version. A PyTorch project that (1) generates Moving MNIST sequences, (2) encodes frames with a frozen CNN to embeddings, (3) runs a sparse reservoir with a low-rank recurrent adapter, and (4) trains the adapter with antithetic ES across 8 GPUs while training the linear decoder with standard supervision.