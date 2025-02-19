## AdapSafe: Adaptive and Safe-Certified Deep Reinforcement Learning-Based Frequency Control for Carbon-neutral Power Systems



### AdapSafe

------

This is the official implementation for [AdapSafe: Adaptive and Safe-Certified Deep Reinforcement Learning-Based Frequency Control for Carbon-neutral Power Systems], which proposes a novel **DRL-based frequency control framework**, AdapSafe, to simultaneously address the two crucial challenges of **safety guarantee and adaptiveness enhancement in a non-stationary environment** to facilitate its real-world applications.

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

### Cite

If you find this resource helpful, please consider starting this repository and cite our research:
 
@inproceedings{wan2023adapsafe,
  title={AdapSafe: adaptive and safe-certified deep reinforcement learning-based frequency control for carbon-neutral power systems},
  author={Wan, Xu and Sun, Mingyang and Chen, Boli and Chu, Zhongda and Teng, Fei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={4},
  pages={5294--5302},
  year={2023}
}
