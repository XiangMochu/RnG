<div align="center">

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

### CVPR 2026 Highlight

<h6>
<a href="https://scholar.google.com/citations?user=OC5oCTgAAAAJ" target="_blank">Mochu Xiang</a><sup>1,2 ‡</sup>,
<a href="https://scholar.google.com/citations?user=pfzhh0IAAAAJ" target="_blank">Zhelun Shen</a><sup>2 ✽ †</sup>,
<a href="https://scholar.google.com/citations?user=HIeMGxcAAAAJ" target="_blank">Xuesong Li</a><sup>3,4</sup>,
<a href="http://npu-cvr.cn/" target="_blank">Jiahui Ren</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=Qa1DMv8AAAAJ" target="_blank">Jing Zhang</a><sup>3</sup>,
<br>
<a href="https://scholar.google.com/citations?user=kWzyOa8AAAAJ" target="_blank">Chen Zhao</a><sup>2</sup>,
<a href="" target="_self"> Shanshan Liu </a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=pnuQ5UsAAAAJ" target="_blank">Haocheng Feng</a><sup>2</sup>,
<a href="https://jingdongwang2017.github.io" target="_blank">Jingdong Wang</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=fddAbqsAAAAJ" target="_blank">Yuchao Dai</a><sup>1 †</sup>
</h6>
<p> <sup>1</sup> Northwestern Polytechnical University, China
<span style="margin: 0 20px"> <sup>2</sup> Baidu Inc., China </span>
<span style="margin: 0 20px"> <sup>3</sup> Australian National University, Australia </span>
<span style="margin: 0 20px"> <sup>4</sup> CSIRO, Australia </span>
</p>

<div align="center">
    <a href="https://npucvr.github.io/RnG/"><strong>Project Page</strong></a> |
    <a href="https://arxiv.org/abs/2603.01194"><strong>Paper</strong></a>  |
    <a href="https://huggingface.co/MochuXiang/RnG/tree/main"><strong>Model</strong></a>  |
    <a href="https://huggingface.co/datasets/MochuXiang/FluffyElephant/tree/main"><strong>Data_44k</strong></a>  |
    <a href="https://huggingface.co/datasets/benzlxs/objaverse_rendering_set/tree/main"><strong>Data_80k</strong></a>
</div>


<img src="https://npucvr.github.io/RnG/imgs/RnG_architecture.png" width="97%"/>

</div>

> **The Network Architecture of RnG.** (a) Source view images are first tokenized using the DINO vision transformer; the Plücker ray map representing the target view point goes through a linear layer. After adding camera tokens for each view, all tokens will then alternately attend to global- and frame-level attention blocks. Finally, camera tokens from input views are used to estimate camera poses, while a point head and an RGB head process ray tokens from the target view, providing geometry and appearance estimations. (b) In inference, the model can cache K/V token from source views, synthesizing novel view geometry and geometry at a higher speed.

## 1. Run the demo

### 1.1 Environment

``` 
conda create -n rng python=3.11
conda activate rng
pip install -r requirements.txt
```

### 1.2 Checkpoints

Download the checkpoint `RnG.pt` from [Huggingface](https://huggingface.co/MochuXiang/RnG/tree/main) and put it under `experiments/checkpoints/RnGUP_Med`.

### 1.3 Evaluation & Demo data

Download the mluti-view renderings of the GSO dataset `gso_render_25v.tar` from [Huggingface](https://huggingface.co/datasets/MochuXiang/FluffyElephant/tree/main), extract it and modify the path in `configs/RnG_obj_medium_bf16_40k.yaml`:
``` 
  val_dataset_cfgs:
    root_dir: /workspace/ssd1/gso_render_rv  # <- modify line 82
    suffix: render_mvs_25/model/             # <- 
```

### 1.4 Launch the demo
```
bash scripts/run_viser_demo.sh
```

## 2. Train the model

We are organizing our training code. Meanwhile, you can checkout the training data [LVIS_subset_44k_25v](https://huggingface.co/datasets/MochuXiang/FluffyElephant/tree/main) and [LGM_subset_80k_40v](https://huggingface.co/datasets/benzlxs/objaverse_rendering_set/tree/main). They contain posed multi-view RGBD renderings of the Objaverse dataset. 