# TRANS
Official implementation for paper "Predictive Modeling with Temporal Graphical Representation on Electronic Health Records"

## Requirements

Requirements and recommended versions:

Python (3.10.13)

pytorch (1.12.1)

torch-geometric (2.3.1)

Pyhealth (1.1.4)

## Data Processing

For MIMIC-III and MIMIC-IV: refer to https://pyhealth.readthedocs.io/en/latest/api/datasets.html; 

For CCAE: Run process_ccae.ipynb in the data folder.


## Training & Evaluation

To train the model and baselines in the paper, run this command:

```
python train.py --model <TRANS/Transformer/...> --dataset <mimic3/mimic4/...>
```

## References
If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{chen2024trans,
  title={Predictive Modeling with Temporal Graphical Representation on Electronic Health Records},
  author={Chen, Jiayuan and Yin, Changchang and Wang, Yuanlong and Zhang, Ping},
  booktitle={International Joint Conference on Artificial  Intelligence},
  year={2024}
}
```
