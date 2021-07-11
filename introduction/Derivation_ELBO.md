### Derivation of ELBO (As Anonymous GitHub could not present a markdown file in well format, we recommend you copy the following content and open it with Typora.)

In this section, we derive the ELBO for VRBot as following, the learning objective from 1-st turn to the $t$-th turn is to maximize the following formula,


$$
\begin{split}
      & \log \prod_{i=1}^{t} p_{\theta}(R_i|R_{i-1}, U_i, G^{global}) \\\\
      = & \log \prod_{i=1}^{t} \sum_{S_i, A_i}  p_{\theta_g}(R_i|S_i, A_{i}, R_{i-1},U_i)
      \cdot p_{\theta_s}(S_{i}) \cdot p_{\theta_a}(A_{i}) \\\\
      = & \log \prod_{i=1}^{t-1} \sum_{S_i, A_i}  p_{\theta_g}(R_i|S_i, A_{i}, R_{i-1},U_i) \cdot p_{\theta_s}(S_{i}) \cdot p_{\theta_a}(A_i) \\\\
      \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t) \cdot p_{\theta_s}(S_{t}) \cdot p_{\theta_a}(A_{t}) \Big] \\\\
      = & \log \prod_{i=1}^{t-1} \sum_{S_i, A_i}  p_{\theta}(R_i| R_{i-1},U_i,G^{global})
      \cdot p_{\theta}(S_{i}, A_{i}|S_{i-1}, R_{i-1}, U_i, G^{global}, R_i) \\\\
      \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
      \cdot p_{\theta_s}(S_{t}) \cdot p_{\theta_a}(A_{t}) \Big] \text{(Bayes' rule)},\\\\
      \end{split}
$$

where $p_{\theta}(S_i, A_i|S_{i-1}, R_{i-1}, U_i, G^{global}, R_i)$ is the joint posterior distribution over $S_i, A_i$, we reach the above transition via Bayes' rule. As we leverage distribution $q_{\phi_s}(S_t) \cdot q_{\phi_a}(A_t)$ to approximate the true posterior distribution, we have the following approximate equality,

$$
\begin{split}
      & \log \prod_{i=1}^{t} p_{\theta}(R_i|R_{i-1}, U_i, G^{global}) \\\\
      \approx & \log \prod_{i=1}^{t-1} \sum_{S_i, A_i}  p_{\theta}(R_i| R_{i-1},U_i, G^{global})
      \cdot q_{\phi_s}(S_{i}) \cdot q_{\phi_a}(A_{i}) \\\\
      \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
      \cdot p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t) \Big],\\\\
  \end{split}
$$



Where $q_{\phi_s}(S_i)$ and $q_{\phi_a}(A_i)$ are the approximate posterior distributions over $S_{i}$ and $A_{i}$, respectively. In the $t$-th turn, $U_{\le {t}}$ and $R_{\le {t-1}}$ are given, thus we omit the marginal distribution $p_{\theta}(R_i|R_{i-1}, U_t, G^{global})$ if $i \le t-1$. We can derive the following formula:

$$
\begin{split}
    & \log \prod_{i=1}^t p_{\theta}(R_i|R_{i-1}, U_{i}, G^{global}) \\\\
    \approx & \log \prod_{i=1}^{t-1} \sum_{S_i, A_i}    p_{\theta}(R_i| R_{i-1},U_i)
    \cdot q_{\phi_s}(S_{i}) \cdot q_{\phi_a}(A_{i}) \\\\
    \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
    \cdot p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t) \Big],\\\\
    = & \log \prod_{i=1}^{t-1} \sum_{S_i, A_i} q_{\phi_s}(S_{i}) \cdot q_{\phi_a}(A_{i}) \\\\
    \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
    \cdot p_{\theta_s}(S_{t})\cdot p_{\theta_a}(A_{t}) \Big] \\\\
    & \text{(}U_{\le t} \text{ and } R_{\le t - 1} \text{ are given)},\\\\
\end{split}
$$
As the state follows the homogeneous Markov hypothesis, which means $t$-th turn state (i.e., $S_t$) only relies on the previous turn state (i.e., $S_{t-1}$). Besides, $A_t$ only relies on $S_t$, then we have,

$$
\begin{split}
    & \log p_{\theta}(R_t|R_{t-1},U_t,G^{global}) \\\\
    = & \log \sum_{S_{t-1}, A_{t-1}} q_{\phi_s}(S_{t-1}) \cdot q_{\phi_a}(A_{t-1}) \\\\
    \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
    \cdot p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t) \Big] \\\\
    & \text{(} p_{\theta_s}(S_t) \text{only relies on } S_{t-1} \text{)} \\\\
    = & \log \sum_{S_{t-1}} q_{\phi_s}(S_{t-1}) \\\\
    \cdot & \Big[ \sum_{S_t, A_t}  p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)
    \cdot  p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t) \Big] \\\\
    & \text{(}p_{\theta_a}(A_{t}) \text{ only relies on } S_{t}\text{)}, \\\\
\end{split}
$$

At the $t$-th turn, following [1], we derive the ELBO as following:

$$
\begin{split}
        & \log p_{\theta}(R_t|R_{t-1}, U_{t}, G^{global}) \\\\
        \approx & E_{q_{\phi_s}(S_{t-1})} \big[ \log E_{p_{\theta_s}(S_t)\cdot p_{\theta_a}(A_t)} [p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)] \big] \\\\
        \ge & E_{q_{\phi_s}(S_{t-1})} \big[ E_{q_{\phi_s}(S_t) \cdot q_{\phi_a}(A_t)}[\log p_{\theta_g}(R_t|R_{t-1}, U_t, S_t, A_t)] \\\\ 
        &  \quad \quad \quad - KL(q_{\phi_s}(S_{t}) \cdot q_{\phi_a}(A_{t})) || p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_{t}) \big], \\\\
        = & - \mathcal{L}_{joint} \\\\
\end{split}
$$

where $q_{\phi_s}(S_{t-1})$ is the abbreviation of $q_{\phi_s}(S_{t-1}|S_{t-2}, R_{t-2}, U_{t-1}, R_{t-1})$. Besides, we only execute $p_{\theta_s}(S_t)$ and $p_{\theta_a}(A_t)$ to infer state $S_t$ and action $A_t$, then send it to $p_{\theta_g}(R_t|\cdot)$ to generate response $R_t$.



### Derivation of 2-stage collapsed inference

In the training stage, we optimize the state tracker and policy network separately instead of optimize all models jointly. We decompose the training process into two stages, i.e., *state optimization stage* and *action optimization stage*. In the state optimization stage, we ignore to estimate the action posterior distribution, and only use the prior policy network in VRBot, then we have,

$$
\begin{split}
  & \log p_{\theta}(R_t| R_{t-1}, U_t, G^{global}) \\\\
  \approx & E_{q_{\phi_s}(S_{t-1})} \big[ \log E_{p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t)} [p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)] \big] \\\\
  = & E_{q_{\phi_s}(S_{t-1})} \Big[ E_{p_{\theta_s}(S_t)} \big[ \log  E_{p_{\theta_a}(A_t)} [p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)] \big] \Big] \\\\
  \ge & E_{q_{\phi_s}(S_{t-1})} \Big[E_{ q_{\phi_s}(S_t)} \big[ \log E_{p_{\theta_a}(A_t)} [ p_{\theta_g}(R_t|S_t, A_t, R_{t-1}, U_t)]\big] \\\\
  & - D_{KL}(q_{\phi_s}(S_t)||p_{\theta_s}(S_t)) \Big] \\\\
  = & E_{q_{\phi_s}(S_{t-1})} \Big[E_{ q_{\phi_s}(S_t)} \big[E_{p_{\theta_a}(A_t)} [ \log p_{\theta_g}(R_t|S_t, A_t, R_{t-1}, U_t)]\big] \\\\ 
  & - D_{KL}(q_{\phi_s}(S_t)||p_{\theta_s}(S_t)) \Big] \\\\
  = & - \mathcal{L}_{s} \\\\
\end{split}
$$

Likewise, in the action optimization stage, we derive the ELBO for approximating the state posterior distribution. Shown as follows:

$$
\begin{split}
  & \log p_{\theta}(R_t| R_{t-1}, U_t, G^{global}) \\\\
  \approx & E_{q_{\phi_s}(S_{t-1})} \big[ \log E_{p_{\theta_s}(S_t) \cdot p_{\theta_a}(A_t)} [p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)] \big] \\\\
  = & E_{q_{\phi_s}(S_{t-1})} \Big[ E_{p_{\theta_s}(S_t)} \big[  E_{p_{\theta_a}(A_t)} [\log p_{\theta_g}(R_t|S_t, A_{t}, R_{t-1},U_t)] \big] \Big] \\\\
  \ge & E_{q_{\phi_s}(S_{t-1})} \Big[E_{ p_{\theta_s}(S_t)} \big[E_{q_{\phi_a}(A_t)} [ \log p_{\theta_g}(R_t|S_t, A_t, R_{t-1}, U_t)] \\\\
  & - D_{KL}(q_{\phi_a}(A_t)||p_{\theta_a}(A_t))\big] \Big] \\\\
  = & - \mathcal{L}_{a} \\\\
\end{split}
$$

[1] Kingma, D. P. and Welling, M. (2013). Auto-encodingvariational bayes.arXiv preprint arXiv:1312.6114.
