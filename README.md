## 基于Q集成下置信域引导的高效扩散策略 &mdash; Official PyTorch Implementation

Efficient Q-Ensemble Diffusion Policy for Offline Reinforcement Learning
zhangxu zhangfeng <br>


Abstract: * Offline reinforcement learning (ORL) often relies on datasets collected by multiple suboptimal or random policies, resulting in complex multimodal action distributions. While diffusion policies can effectively model such distributions and mitigate distributional shift, they suffer from high inference latency due to multi-step reverse diffusion and may still overestimate uncertain out-of-distribution (OOD) actions. To address these limitations, we propose the Efficient Q-Ensemble Diffusion Policy (E2DP). E2DP introduces a two-step reverse inference mechanism that simplifies the standard multi-step diffusion process while maintaining strong multimodal modeling capability. In addition, a lower-confidence-bound (LCB) guided Q-ensemble is designed to enhance estimation reliability by applying independent targets and variance regularization. This allows E2DP to suppress overestimation of uncertain actions using only a small number of Q-networks. Experiments on Bandit and D4RL benchmarks demonstrate that E2DP achieves comparable expressiveness to existing diffusion-based methods while reducing inference latency by approximately 2.5¡Á and attaining the best normalized scores across most tasks, verifying its efficiency and robustness.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.

### Running
Running experiments based our code could be quite easy, so below we use `walker2d-medium-expert-v2` dataset as an example. 

For reproducing the optimal results, we recommend running with 'online model selection' as follows. 
The best_score will be stored in the `best_score_online.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay
```

For conducting 'offline model selection', run the code below. The best_score will be stored in the `best_score_offline.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms offline --lr_decay --early_stop
```

Hyperparameters for E2DP have been hard coded in `main.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 

## Citation

If you find this open source release useful, please cite in your paper:
```


