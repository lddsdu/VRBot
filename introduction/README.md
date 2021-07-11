## VRBot: Few-Shot Variational Reasoning for Medical Dialogue Generation

- **Train**

```shell
$ python main.py --task meddg --model vrbot --train_batch_size 16 --test_batch_size 16 --init_tau 3.0 --worker_num 5 --gen_strategy gru --train_stage natural --decay_interval 10000 --init_lr 1e-4 --supervision_rate 0.5 --ablation vrbot --hop 2 --state_num 10 --action_num 3 --auto_regressive
```

- **Test**

```shell
$ python main.py --task meddg --model vrbot --train_batch_size 16 --test_batch_size 16 --init_tau 3.0 --worker_num 5 --gen_strategy gru --train_stage natural --decay_interval 10000 --init_lr 1e-4 --supervision_rate 0.5 --ablation vrbot --hop 2 --state_num 10 --action_num 3 --auto_regressive --test --ckpt xxx.ckpt
```

- **Parameters analysis**

```
--task kamed/meddialog/meddg
--model vrbot/vrbot_bert
--train_stage natural/stage/action
--worker_num using 5 cpus to load data
--gen_strategy default gru, you could also choice mlp
--decay_interval learing rate decay per 10000 learning steps
--init_lr initial learning rate
--supervision_rate 0.5 supervision rate 0.0/0.1/0.25/0.5/1.0
--ablation vrbot/wo-st/wo-pl/wo-ginfer/wo-dinfer (corresponds to VRBot,VRBot\S,VRBot\A,VRBot\g,VRBot\c respectively)
--hop knowledge base search 2-hops
--state_num default 10 which denotes the state span length
--action_num default 3 which denotes the action span length
--auto_regressive whether to autoregressive decoding state & action
--test do test
--ckpt checkpoint file
--device choose gpu x, if use cpu, remove this item
```

- **Dataset**
Note: We anonymed all the datasets, removing information that contained the names, contact details, and organizations of patients and doctors. As Anonymous GitHub could not present .zip file as it is too big to be anonymized, we will share the dataset upon publication of the paper.
The dataset will be shared upon publication of the paper
```
KaMed: kamed_train.zip kamed_valid.zip kamed_test.zip
MedDialog: meddialog_train.zip meddialog_valid.zip meddialog_test.zip
MeDDG: meddg_train.zip meddg_valid.zip meddg_test.zip
```


Besides, readers may interested in how can we derive the learning objective $\mathcal{L}_{joint}, \mathcal{L}_{s}, \mathcal{L}_a$, we provide the detail derivation of ELBO and a detail generation demonstration, please refer to [Derivation_ELBO.md](Derivation_ELBO.md) for more details.
