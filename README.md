## AdapSafe: Adaptive and Safe-Certified Deep Reinforcement Learning-Based Frequency Control for Carbon-neutral Power Systems



### AdapSafe

------

This paper proposes a novel **DRL-based frequency control framework**, AdapSafe, to simultaneously address the two crucial challenges of **safety guarantee and adaptiveness enhancement in a non-stationary environment** to facilitate its real-world implementation. In particular, a self-tuning CBF-based compensator is designed to realize the optimal safety compensation under different risk conditions, which greatly reduces the control cost. Furthermore, to minimize the control cost without sacrificing safety, the CBF-based safe control method is integrated with meta reinforcement learning algorithm with the innovations of transition post-procession and noise elimination scheme to achieve safety assurance during the meta-training phase for multi-task scenarios. Through a comparative study with the state-of-the-art frequency control methods based on **a GB 2030 power system**, the results demonstrate that AdapSafe can achieve the target of safety guarantee under a variational environment with superior control performance for both the training and testing phases.

### Environment

------

The AdapSafe method is implemented in a simplified model of system frequency dynamics as shown in Figure 1. For the VSM-based control scheme, $T_g$ and $R_g$ represent the turbine time constant and drop coefficient, respectively. $T_c$ is the converter time constant, while $M_c$ and $D_c$ represent the virtual inertia and damping constant of the converter. Meanwhile, for the DRL-based control scheme, the frequency adjustment $\Delta P_v$ brought by RES will be determined by its policy network $\pi(\Delta f, \Delta P_g;\theta)$.

![Appendix_Dynamics](readme/Appendix_Dynamics.png)

Figure 1. Simplified system frequency dynamics model.

### Getting Started

------



```python
pip install -r requirements.txt
```

### Train

------



```python
# For TD3
python main.py
# For MQL
python main.py --enable_adaptation --enable_context
# For CBF-TD3
python main.py --enable_CBF
# For AdapSafe
python main.py --enable_CBF --enable_adaptation --enable_context
```

For more other parameter settings, please refer to "ArgumentParser" in our code.

### Test

------



```python
# For single test
python main.py --test --max_test_episode 4
# For multi test
python main.py --multi_test --test_tasks 50
```

