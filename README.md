<div align="center">

# RnG: A Unified Transformer for Complete 3D Modeling from Partial Observations

### CVPR 2026

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

🚧 *We are currently organizing our code* 🚧

<img src="https://npucvr.github.io/RnG/imgs/RnG_architecture.png" width="97%"/>

> **The Network Architecture of RnG.** (a) Source view images are first tokenized using the DINO vision transformer; the Plücker ray map representing the target view point goes through a linear layer. After adding camera tokens for each view, all tokens will then alternately attend to global- and frame-level attention blocks. Finally, camera tokens from input views are used to estimate camera poses, while a point head and an RGB head process ray tokens from the target view, providing geometry and appearance estimations. (b) In inference, the model can cache K/V token from source views, synthesizing novel view geometry and geometry at a higher speed.