# MuSe2021 || <a href="https://www.muse-challenge.org/">HOMEPAGE</a>

## Baseline model: LSTM Regressor / Classifier 

Sub-challenges: **MuSe-Wilder** and **MuSe-Stress** which focus on continuous emotion (valence and arousal) prediction; **MuSe-Sent**, in which participants recognise five classes each for valence and arousal; and **MuSe-Physio**, in which the novel aspect of physiological-emotion is to be predicted. 

Our baseline results on test are: <br />
* **MuSe-Wilder** CCC: **.5974** for continuous valence using late fusion; CCC: **.3386** for continuous arousal using DeepSpectrum features <br />
* **MuSe-Sent** F1 score: **32.91%** for valence utilising late fusion of vision and text; F1: **35.12%** for arousal utilising a late fusion of audio-video features <br />
* **MuSe-Stress** CCC: **.5058** for valence based on VGGface features; CCC: **.4474** for arousal based one GeMAPS features <br />
* **MuSe-Physio** CCC: **.4606** for physiological-emotion prediction <br />

<a href="https://www.researchgate.net/publication/350874530_The_MuSe_2021_Multimodal_Sentiment_Analysis_Challenge_Sentiment_Emotion_Physiological-Emotion_and_Stress">Details</a>

For this years’ challenge, we utilise the **MuSe-CaR** dataset focusing on user-generated reviews and introduce the **Ulm-TSST** dataset, which displays people in stressful depositions. 

## Installation 
Create a virtenv; check and install packages in requirements.txt; download the data; adjust paths in `config.py`; run sample calls. 

## Run: Sample calls for each sub-challenges 

### MuSe-Wilder:

```
python main.py --task wilder --emo_dim arousal --feature_set egemaps --normalize --norm_opts y --d_rnn 128 --rnn_n_layers 4 --epochs 100 --batch_size 1024 --n_seeds 10 --win_len 200 --hop_len 100 --use_gpu --cache --save --save_path preds
```

### MuSe-Sent:

```
python main.py --task sent --emo_dim arousal --feature_set bert --d_rnn 64 --rnn_n_layers 4  --lr 0.0001 --epochs 100 --batch_size 1024 --n_seeds 10 --win_len 200 --hop_len 100 --use_gpu --cache --save --save_path preds
```

### MuSe-Stress: 

```
python main.py --task stress --emo_dim arousal --feature_set vggface --d_rnn 64 --rnn_n_layers 4 --rnn_bi --lr 0.002 --epochs 100 --batch_size 1024 --n_seeds 10 --win_len 300 --hop_len 50 --use_gpu --cache --save --save_path preds
```

### MuSe-Physio: 

```
python main.py --task physio --emo_dim anno12_EDA --feature_set vggish --d_rnn 32 --rnn_n_layers 2 --rnn_bi --lr 0.005 --epochs 100 --batch_size 1024 --n_seeds 10 --win_len 300 --hop_len 50 --use_gpu --cache --save --save_path preds
```

### Late Fusion:
Set `--save` in the above calls to save predictions for data samples of all partitions for late fusion.

```
python late_fusion.py --task wilder --emo_dim valence --preds_path preds --d_rnn 32 --rnn_n_layers 2 --epochs 20 --batch_size 64 --lr 0.001 --n_seeds 5 --win_len 200 --hop_len 100 --use_gpu --predict
```

## Settings
See main.py for more argparser options. 
#### GPU:
`--use_gpu`: We highly recommend to execute the training on a GPU machine. This moves the tensors and models to cuda in Pytorch. 

#### Data segmentation:
`--win_len X`: Specify the window length for each segment.<br />
`--hop_len X`: Specify the hop length for each segment.<br />

#### Normalisation:
`--normalize`: Specify if any features should be normalized. <br />
`--norm_opts y,n`: In case for early fusion. Specify which feature in the list has to be normalised ("y": yes, "n": no) in the corresponding order to the feature_set.<br />

#### Training:
`--n_seeds 5`: Specify number of random seeds to try for the hyperparameters (same settings, multiple runs). This helps to rule out a bad local minima but increases computation by factor of X n_seeds.<br />
`--cache`: Training can be done faster if the pre-processed data is kept. If you add new features, remove the chached data! The same applies to data augmentation before the data pipeline.<br />

#### Early fusion:
Just adding multiple features in the feature_set parameter like `--feature_set vggface bert vggface`. Our late fusion results were superior to the early fusion with this network architecture. Therefore, only late fusion is reported. However, nothing wrong with testing early fusion more comprehensively.:)

#### Late fusion:
Set option `--predict` for masked test labels. The predictions for all samples in the test partition are saved in csv files. 


## Reproducibility 
We cannot guarantee perfectly reproducible results even when you use identical seeds due to changing initializations across PyTorch releases, CPU/GPU use, or different platforms (see https://pytorch.org/docs/stable/notes/randomness.html). However, we did our best and added every gadget that improved reproducibility and did not dramatically slow down training performance. Using the HP in the paper, you will come into the same result cooridor. Furthermore, many weights of the best models can be found here: https://drive.google.com/drive/folders/14mKL4uxRTGeK16ViDFYNgxfOdI27hEBa?usp=sharing


## Citation
We have a bunch of papers you can cite related to this baseline libary:

MuSe2021 - preprint:
```bibtex
@inproceedings{stappen2021muse,
  title={The MuSe 2021 Multimodal Sentiment Analysis Challenge: Sentiment, Emotion, Physiological-Emotion, and Stress},
  author={Stappen, Lukas and Baird, Alice and Christ, Lukas and Schumann, Lea and Sertolli, Benjamin and Messner, Eva-Maria and Cambria, Erik and Zhao, Guoying and Schuller, Bjoern W},
  booktitle={Proceedings of the 2nd International Multimodal Sentiment Analysis Challenge and Workshop},
  year={2021}
}
```

Dataset:
```bibtex
@article{stappen2020dataset,
	title        = {The Multimodal Sentiment Analysis in Car Reviews (MuSe-CaR) Dataset: Collection, Insights and Improvements},
	author       = {Lukas Stappen and Alice Baird and Lea Schumann and Björn Schuller},
	year         = 2021,
	month        = {06},
	journal      = {IEEE Transactions on Affective Computing},
	publisher    = {IEEE Computer Society},
	address      = {Los Alamitos, CA, USA},
	number       = {01},
	pages        = {1--16},
	doi          = {10.1109/TAFFC.2021.3097002},
	issn         = {1949-3045},
	keywords     = {sentiment analysis;annotations;task analysis;databases;affective computing;social networking (online);computational modeling}
}
```

MuSe2020:
```bibtex
@inproceedings{stappen2020muse,
  title={MuSe 2020 Challenge and Workshop: Multimodal Sentiment Analysis, Emotion-target Engagement and Trustworthiness Detection in Real-life Media: Emotional Car Reviews in-the-wild},
  author={Stappen, Lukas and Baird, Alice and Rizos, Georgios and Tzirakis, Panagiotis and Du, Xinchen and Hafner, Felix and Schumann, Lea and Mallol-Ragolta, Adria and Schuller, Bjoern W and Lefter, Iulia and others},
  booktitle={Proceedings of the 1st International on Multimodal Sentiment Analysis in Real-life Media Challenge and Workshop},
  pages={35--44},
  year={2020}
}
```

## Acknowledgement & Contributors ✨ : 
We were inspired by our last year's baseline models (https://github.com/lstappen/MuSe2020) and some of the winners to create this baseline code. Thanks to all who contributed, especially:

<table>
  <tr>
    <td align="center">
<a href="https://github.com/leaschumann"><img src="https://avatars.githubusercontent.com/u/28183944?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lea</b></sub></a><br /><td align="center">
<a href="https://github.com/lc0197"><img src="https://avatars.githubusercontent.com/u/44441963?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Lukas</b></sub></a><br /><td align="center">
<a href="https://github.com/benni-ser"><img src="https://avatars.githubusercontent.com/u/28057187?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Benni</b></sub></a><br /><td align="center">
<a href="https://github.com/aliceebaird"><img src="https://avatars.githubusercontent.com/u/10690171?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Alice</b></sub></a><br />
  </tr>
</table>

If you like what you see, please leave us a ✨!
