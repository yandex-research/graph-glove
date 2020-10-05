# GraphGlove
Supplementary code for the EMNLP 2020 paper "Embedding Words in Non-Vector Space with Unsupervised Graph Learning".

# What does it do?
It learns to represent words as a graph with learned weights and topology. The resulting graph outperforms vector-based embeddings (GloVe and Poincare Glove) on a set of word similarity and analogy benchmarks.

# What do I need to run it?
* A machine with a lot of CPU cores (preferably 8+)
  * Our implementation does not support GPU but can run on multiple CPU machines
* A popular Linux x64 distribution
  * Tested on Ubuntu16.04, Ubuntu 18.04, should work fine on any popular linux64 and even MacOS;
  * Windows and x32 systems may require heavy wizardry to run;
  * When in doubt, use Docker, e.g. [this one](https://hub.docker.com/r/pytorch/pytorch/).


# Environment setup
1. Clone or download this repo. `cd` to its root directory.
2. Grab or build a working python enviromnent. [Anaconda](https://www.anaconda.com/) works fine.
3. ```sudo apt-get install redis-server build-essential gcc g++ swig gcc-4.8 g++-4.8```
   * If you don't have superuser privelegies, redis-server can be [installed locally](https://techmonger.github.io/40/redis-without-root/);
4. Install packages from `requirements.txt`. All but three can be installed with pip
  * `faiss` is installed using the [official installation guide](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md);
  * `torch` *can* be installed with pip, but we recommend [this  page](https://pytorch.org/get-started/locally/) for more options;
  * `glove-python` installs normally on python3.6; For >=3.7 use [this workaround](https://github.com/maciejkula/glove-python/issues/96#issuecomment-499840851);

You're now ready to run the code!

### Training the model
Training runs in two processes:
* `scripts/run_master.py` - builds the model, does the optimization, evaluates metrics - basically runs everything except for pathfinding;
* `scripts/run_worker.py` - periodically loads the model and finds shortest paths for training batches, sends results back to master;

To run locally, first download wiki17 50k co-occurrences:
```bash
# assuming you are in repository root folder
mkdir -p word_embeddings_50k
cd word_embeddings_50k
wget https://www.dropbox.com/s/rub6334e2cuxcar/init_glove_50k_dot.npz?dl=1 -O init_glove.npz
wget https://www.dropbox.com/s/7wrhagpz4elk200/cooc_50k.pkl?dl=1 -O cooc.pkl
cd -

# If you want to brew your own corpora, use ./scripts/data_preprocessing.py
```


Then run the master script
```bash
python scripts/run_master.py --port 6999 --password securepassword123 --batch_size 64 --buffer_size_multiple 2 \
  --data_dir word_embeddings_50k --report_frequency 250 --lr 0.01 --exp_name word2graph_simple_50k \
  --soft --clip_prob_max 0.99 --knn_edges 64 --random_edges 10 --base_lambda 5 --lambda_warmup 5000
```

Finally, run the worker script
```bash
python3 scripts/run_worker.py --host localhost --port 6999 --password securepassword123 --n_threads -1 --restart_on_error
```

The worker is designed to wait until master begins training. If it has not happened yet, worker will periodically print error messages and restart.
You can set the datasets for word similarity evaluation with --track-benchmarks. For word analogy evaluation, refer 

__Distributed training:__ you can train faster by using multiple CPU machines
* first, run master process on the most powerful machine
* then, one "worker" thread on each subsequent machine. Change ```--host localhost``` to master's host

This setup assumes that workers can access master on port 6999. If not, either pick a different port or use ssh [port forwarding](https://www.ssh.com/ssh/tunneling/example). 

Training does not scale infinitely: once you find that trainer "receives a new batch" in less than 0.05 seconds, it means you've hit the worker cap.

### Play around with trained embeddings
You can explore the trained embeddings using [this notebook](./scripts/demo_evaluate.ipynb). By default, it works with a pre-trained model that was obtained on 50K most frequent tokens from wikipedia 2017 with distance-based loss. However, you can seamlessly replace that pre-trained model with the one you trained in the previous section.

### Contacts
For any issues with running the code or questions about our work, please create an issue or contact us [by email](mryabinin@hse.ru).

### References
If you find this repository useful, please cite the paper:
```
@inproceedings{ryabinin2020embedding,
  title={Embedding Words in Non-Vector Space with Unsupervised Graph Learning},
  author={Ryabinin, Max and Popov, Sergei and Prokhorenkova, Liudmila and Voita, Elena},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2020}
}
```