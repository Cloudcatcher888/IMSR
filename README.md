# IMSR

## Introduction 
In recent years, sequential recommendation has been widely researched, which aims to predict the next item of interest based on user’s previously interacted item sequence. Existing works utilize capsule network and self-attention method to explicitly capture multiple underlying interests from a user’s interaction sequence, achieving the state-of-the-art sequential recommendation performance. In practice, the lengths of user interaction sequences are ever-increasing and users might develop new interests from new interactions, and a model should be updated or even expanded continuously to capture the new user interests. We refer to this problem as incremental multiinterest sequential recommendation, which has not yet been well investigated in the existing literature. In this paper, we propose an effective incremental learning framework for multi-interest sequential recommendation called IMSR, which augments the traditional fine-tuning strategy with the existing-interests retainer (EIR), new-interests detector (NID), and projection-based interests trimmer (PIT) to adaptively expand the model to accommodate user’s new interests and prevent it from forgetting user’s existing interests. Extensive experiments on real-world datasets verify the effectiveness of the proposed IMSR on incremental multi-interest sequential recommendation, compared with various baseline approaches.

## Architecture

![](/arch.png)


## Requirement

```
pytorch == 1.14
python == 3.7
```

## Instruction
1, You can run the code by: 

```
python code/IMSR.py
```

3, You can change customize the initial interest number K and \delta K in utils.Config.

4, result on Electronic:

![](/Electronic-NDCG.png)

If you use this code, please cite our paper as follows:
```
Zhikai Wang, Yanyan Shen, Incremental Learning for Multi-interest Sequential Recommendation, in IEEE ICDE, 1071-1083
```
