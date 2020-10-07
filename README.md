# NeurIPS 2020: MineRL Competition SQIL Baseline with PFRL

This repository is a **[soft Q imitation learning (SQIL)](https://arxiv.org/abs/1905.11108) baseline submission example with [PFRL](https://github.com/pfnet/pfrl)**,
based on [the MineRL Rainbow Baseline with PFRL](https://github.com/keisuke-nakata/minerl2020_submission).

For detailed & latest documentation about the competition/template, see the original template repository.

This repository is a sample of the "Round 1" submission, i.e., the agents are trained locally.  

`test.py` is the entrypoint script for Round 1.  
Please ignore `train.py`, which will be used in Round 2.

`train/` directory contains baseline agent's model weight files trained on `MineRLObtainDiamondVectorObf-v0`.

## List of current baselines
- [Rainbow](https://github.com/keisuke-nakata/minerl2020_submission)
- **SQIL** <-- We are here
- [DQfD](https://github.com/marioyc/minerl2020_dqfd_submission)
- [PDDDQN](https://github.com/ummavi/minerl2020_submission)

# How to Submit

After [signing up the competition](https://www.aicrowd.com/challenges/neurips-2020-minerl-competition), specify your account data in `aicrowd.json`.
See [the official doc](https://github.com/minerllabs/competition_submission_template#what-should-my-code-structure-be-like-)
for detailed information.

Then you can create a submission by making a tag push to your repository on https://gitlab.aicrowd.com/. Any tag push (where the tag name begins with "submission-") to your repository is considered as a submission.

![](https://i.imgur.com/FqScw4m.png)

If everything works out correctly, you should be able to see your score on the
[competition leaderboard](https://www.aicrowd.com/challenges/neurips-2020-minerl-competition/leaderboards).

![MineRL Leaderboard](assets/minerl-leaderboard.png)


# About Baseline Algorithm

This baseline consists of three main steps:

1. [Apply K-means clustering](https://minerl.io/docs/tutorials/k-means.html) for the action space with the demonstration dataset.
2. Calculate cumulative reward boundaries for each subtask so that the amount of frames in the demonstration is equally separated.
3. Apply SQIL algorithm on the discretized action space.

K-means in the step 1 is from [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans).
In this baseline, **the agent maintains two clusters with different sampling criteria**: One is sampled from frames changing `vector` in the next observation and the other is from the remaining.

The implementation of SQIL is not included into [PFRL](https://github.com/pfnet/pfrl) agents but it is based on PFRL's DQN implementation.


# How to Train Baseline Agent on your own

`mod/` directory contains all you need to train agent locally:

```bash
pip install numpy scipy scikit-learn pandas tqdm joblib pfrl

# Don't forget to set this!
export MINERL_DATA_ROOT=<directory you want to store demonstration dataset>


python3 mod/sqil.py \
  --gpu 0 --env "MineRLObtainDiamondVectorObf-v0"  \
  --outdir result \
  --replay-capacity 300000 --replay-start-size 5000 --target-update-interval 10000 \
  --num-step-return 1 --lr 0.0000625 --adam-eps 0.00015 --frame-stack 4 --frame-skip 4 \
  --gamma 0.99 --batch-accumulator mean --exp-reward-scale 10 --logging-level 20 \
  --steps 4000000 --eval-n-runs 20 --arch dueling --dual-kmeans --kmeans-n-clusters-vc 60 --option-n-groups 10
```
or you can call a fixed setting from `train.py`.


# Team

The quick-start kit was authored by 
**[Shivam Khandelwal](https://twitter.com/skbly7)** with help from [William H. Guss](http://wguss.ml)

The competition is organized by the following team:

* [William H. Guss](http://wguss.ml) (Carnegie Mellon University)
* Mario Ynocente Castro (Preferred Networks)
* Cayden Codel (Carnegie Mellon University)
* Katja Hofmann (Microsoft Research)
* Brandon Houghton (Carnegie Mellon University)
* Noboru Kuno (Microsoft Research)
* Crissman Loomis (Preferred Networks)
* Keisuke Nakata (Preferred Networks)
* Stephanie Milani (University of Maryland, Baltimore County and Carnegie Mellon University)
* Sharada Mohanty (AIcrowd)
* Diego Perez Liebana (Queen Mary University of London)
* Ruslan Salakhutdinov (Carnegie Mellon University)
* Shinya Shiroshita (Preferred Networks)
* Nicholay Topin (Carnegie Mellon University)
* Avinash Ummadisingu (Preferred Networks)
* Manuela Veloso (Carnegie Mellon University)
* Phillip Wang (Carnegie Mellon University)


<img src="https://d3000t1r8yrm6n.cloudfront.net/images/challenge_partners/image_file/35/CMU_wordmark_1500px-min.png" width="50%"> 

  <img src="https://d3000t1r8yrm6n.cloudfront.net/images/challenge_partners/image_file/34/MSFT_logo_rgb_C-Gray.png" width="20%" style="margin-top:10px">

 <img src="https://raw.githubusercontent.com/AIcrowd/AIcrowd/master/app/assets/images/misc/aicrowd-horizontal.png" width="20%">   <img src="https://d3000t1r8yrm6n.cloudfront.net/images/challenge_partners/image_file/38/PFN_logo.png" width="15%" style="margin-top:10px">
