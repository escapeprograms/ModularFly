A Universal Abstract Algorithm of Modular Addition, and the Fly

Context:
[1] Clock and Pizza: https://arxiv.org/pdf/2306.17844, https://arxiv.org/pdf/2301.05217 
Transformers trained on the task of modular addition use a “clock” algorithm or “pizza” algorithm

[2] “Uncovering a Universal Abstract Algorithm for Modular Addition in Neural Networks” https://arxiv.org/pdf/2505.18266 
Pizza and Clock are the same underlying algorithm: approximate Chinese Remainder Theorem (aCRT)
This paper shows that this universality hypothesis holds for both transformers and deep NNs

[3] https://flywire.ai/ 
FlyWire v783: a female fly connectome
At each time step, active neurons send excitatory or inhibitory signals to their connections, and all activity levels decay.  We take a reading at an tuned time step that can't be too early (calculation not done) or too late (signal fades)
This is similar to a leaky RNN

[4] Frozen Randomly Initialized Transformer: https://proceedings.neurips.cc/paper_files/paper/2024/file/bccdd196d798a51a4961989984a9ed4a-Paper-Conference.pdf 
We can train the input/output embeddings with a randomly initialized transformer model on toy tasks and get good results.
One such task is modular addition.  The solution it found used a “clock” algorithm, supporting universality
Same setup as this project’s experiment

[5] “From Synapses to Dynamics: Obtaining Function from Structure in a Connectome Constrained Model of the Head Direction Circuit” https://www.biorxiv.org/content/10.1101/2025.05.26.655406v1.full
This study on the fly connectome shows the presence of a structure called a “ring attractor”
Ring attractors are a circular wiring of neurons with weights (synapse counts) that tend to have a clump of activity around a particular neuron.
The clump moves around in accordance to certain signals, such as the fly changing its direction
This behavior can potentially be repurposed into a clock-like mechanism for solving modular addition -> my initial hypothesis

[6] “A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks” https://arxiv.org/html/2610.10014v1 
A trained fly connectome can learn addition for numbers under 100

Research Question 0: Can the fly architecture solve modular addition with trained weights?

Hypothesis: Probably. [6]

Research Question 1: Does this universality hypothesis also hold for a network architecture not designed by a human, i.e. the fly connectome?

Hypothesis: The fly might use its natural circular wiring to implement a clock or pizza algorithm.

Experiment 1:
Train the fly connectome weights and embeddings/unembeddings but keep the connectome architecture.  We start from random weights for the fly, not its natural weights.

Notes:
This setup is meant to extend the setup of [2], where we train the weights directly of different architectures.  The embedding dim of [2] is 128, but for the fly I plan to connect the input embeddings to the fly’s natural sensory neurons and the output embeddings to the fly’s output neurons.


Research Question 2: Are the weights of the fly itself able to solve the modular addition problem?  Will it implement aCRT?  Will it use the same mechanisms as the trainable fly?

Hypothesis: The ring attractors are working and fully functional in real life, so if the fly can implement it in the fully trainable version, I think it can also implement it with the frozen version.

Experiment 2:
Freeze the fly connectome’s natural weights (defined by synapse count), and train only embedding/unembeddings.

Notes: This was actually my original idea in this line of project idea iterations, and I was going to compare it to the frozen transformer in [4].  According to [5], sometimes the weights might not be stable enough to form a ring attractor, and they needed to change the gain (blanket weight multiplier) per different cell type.  This can be a sub-experiment in the case that this doesn’t work out of the box.


Evaluation: aCRT detection
I haven’t looked into the specifics much, but I will probably use the methodology of [2].
