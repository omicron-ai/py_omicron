# End-to-End Trainable Non-Collaborative Dialog System
This branch contains code and dataset for the paper [End-to-End Trainable Non-Collaborative Dialog System](https://arxiv.org/abs/1911.10742), published in AAAI 2020.

Training for AntiScam:
```bash
python3 ./train.py --gradient_accumulation_steps=4 --lm_coef=2.0 --max_history=2 --n_epochs=10 --num_candidates=4 --train_batch_size=2 --valid_batch_size=1 --model_checkpoint="ff" --model="gpt" --da_coef=1.0 --se_coef=1.0
```

Then you can run `interact.py` to interact with your bot:
```bash
python3 ./interact.py --model models/
```

If you use the datasets or any source codes included in this repository in your
work, please cite the following paper:

    @misc{li2019endtoend,
    title={End-to-End Trainable Non-Collaborative Dialog System},
    author={Yu Li and Kun Qian and Weiyan Shi and Zhou Yu},
    year={2019},
    eprint={1911.10742},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
	}   
            
### Dataset
The dataset AntiScam is in /data. There are two files under data/, 1) data/AntiScam_annotated, which contains 93 annotated dialogs, 2) data/AntiScam_all, which contains all the 219 dialogs.









