---
typora-root-url: ../IMSR
---

# IMSR

## Introduction 
This is the official implementation for paper "Incremental Learning for Multi-interest Sequential Recommendation", Zhikai, Wang and Yanyan, Shen, in ICDE 2023 (Best Paper Award).

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
