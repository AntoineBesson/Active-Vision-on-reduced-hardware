# Related work ledger: what to cite and how we differ

One entry per paper that a reviewer could put next to this study. The TRIES
ledgers record what we tried; this file records what others published. Add an
entry whenever a paper is found, even a damaging one.

**Our position, to hold against every entry.** The backbone is frozen
(PIDNet-S, BiSeNetV2, public Cityscapes weights): nothing is trained through
it. Every method is judged against the *same model on the whole frame at a
lower scale*, at equal measured latency on a T4 (mean and p95), on video
(Cityscapes-VPS). Negative results are reported. Most papers below train the
network with their module and compare with a slower network or with FLOPs.

**Status of each entry** (column "Checked"):
`abstract` = title, venue and abstract confirmed by a web search on the date
given; `memory` = written from memory, to confirm before citing; `read` = the
paper itself was read. No entry is `read` yet, so **no number from these
papers may be quoted** until it is.

## Non-uniform resampling of the input (closest to Method 5)

| Paper | What they do | Difference to state | Checked |
|---|---|---|---|
| Thavamani, Li, Ferroni, Ramanan. **Learning to Zoom and Unzoom.** CVPR 2023. arXiv 2303.15390 | Zoom the input with a saliency warp, run the network, unzoom the features with a piecewise bilinear inverse. Detection, **semantic segmentation on Cityscapes**, 3D detection. | **The closest prior work.** They unzoom features inside a network trained with the warp; we unwarp the logits of a frozen network. We compare at equal measured latency against the same network at a lower scale, with an oracle saliency first (a bound), on video. To read in full: what backbone, whether trained, what saliency, and their latency accounting. | abstract, 2026-11-16 |
| Thavamani, Li, Cebron, Ramanan. **FOVEA: Foveated Image Magnification for Autonomous Navigation.** ICCV 2021. arXiv 2108.12102 | Magnify salient regions before a detector; saliency from dataset priors or from the previous frame's detections; anti-cropping constraint. Argoverse-HD, BDD100K. | Detection, trained detector. Their "previous frame" saliency is our `prev` timing; their dataset prior is point 2 of our next research points. To check: whether their warp is separable like ours. | abstract, 2026-11-16 |
| Recasens, Kellnhofer, Stent, Matusik, Torralba. **Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks.** ECCV 2018. arXiv 1809.03355 | Differentiable layer that resamples the input around salient regions with a kernel; trained end to end. Gaze estimation, fine-grained classification. | Origin of the saliency-sampling formula. Not dense prediction (no inverse warp), network trained. Source of the non-separable warp we plan to test. | abstract, 2026-11-16 |
| Ghosh, Reddy, Mertz, Narasimhan. **Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection.** CVPR 2023. arXiv 2303.14311 | Resampling prior from scene geometry (ground plane and a plane above, vanishing point), for detection from traffic and car cameras. | Detection, learned with the detector. A fixed geometric prior needs no per-frame saliency: our planned "static prior" test. | abstract, 2026-11-16 |
| Marin, He, Vajda, Chatterjee, Tsai, Yang, Boykov. **Efficient Segmentation: Learning Downsampling Near Semantic Boundaries.** ICCV 2019 | Content-adaptive downsampling that puts sampling points near semantic boundaries, for segmentation. | Very close to our boundary saliency (B). They train a network to predict the sampling and train the segmenter on it; ours is an oracle on a frozen model. Our chunk 0 finds B weaker than small-class saliency for PIDNet-S: worth comparing. | abstract, 2026-11-16 |
| Jin, Tanno, Mertzanidou, Panagiotaki, Alexander. **Learning to Downsample for Segmentation of Ultra-High Resolution Images.** ICLR 2022. arXiv 2109.11071 | Learnable downsampling module trained jointly with the segmenter; a regulariser keeps it from collapsing on trivial regions. Street-view, aerial, medical. | Trained jointly; images, not video. Their regulariser is the trained counterpart of our clip on the density. | abstract, 2026-11-16 |

## Crops and glimpses over a cheap context (Method 2, and the glimpses of Method 4)

| Paper | What they do | Difference to state | Checked |
|---|---|---|---|
| Chai. **Patchwork: A Patch-wise Attention Network for Efficient Object Detection and Segmentation in Video Streams.** ICCV 2019. arXiv 1904.01784 | Processes only a small window of each frame, memory cells carry context from past frames, a Q-learned policy picks the next window. ImageNet VID, DAVIS. | **The closest prior work to Method 4** (window + memory + learned policy). Trained end to end, object-centric tasks. Our Method 4 bound says that with a frozen real-time segmenter on Cityscapes-VPS even an oracle window selector with a canvas stays below the full frame at equal latency. To read: their baseline (is it the same network at lower resolution?) and latency accounting. | abstract, 2026-11-16 |
| Chen, Jiang, Wang, Cui, Qian. **Collaborative Global-Local Networks for Memory-Efficient Segmentation of Ultra-High Resolution Images (GLNet).** CVPR 2019. arXiv 1905.06368 | A global branch on the downscaled image and a local branch on full-resolution crops, fused. Aerial and medical images of up to 30 M pixels. | Same context + crops structure as Method 2, but trained, on images far larger than 2 MP, and the goal is memory. Supports our reading that crops pay only when the full frame cannot be run at all. | abstract, 2026-11-16 |
| Mnih, Heess, Graves, Kavukcuoglu. **Recurrent Models of Visual Attention.** NeurIPS 2014 | Glimpse network with a recurrent state and a policy trained by reinforcement. | Origin of glimpse-based active vision; classification, trained. Cite as the idea, not as a baseline. | memory |

## Reusing computation across video frames (Method 3)

| Paper | What they do | Difference to state | Checked |
|---|---|---|---|
| Xu, Fu, Yang, Lee. **Dynamic Video Segmentation Network (DVSNet).** CVPR 2018. arXiv 1804.00931 | Splits the frame into regions; a decision network sends each region either to the segmentation network or to a flow network that warps the previous result. Cityscapes. | **The closest prior work to Method 3** (per-region recompute or reuse). They warp with learned optical flow and train a decision network, on a heavy segmenter; we gate with a camera-motion homography and a fixed rule, on real-time segmenters, where the full frame is already cheap. Our negative result is about that regime. | abstract, 2026-11-16 |
| Zhu, Xiong, Dai, Yuan, Wei. **Deep Feature Flow for Video Recognition.** CVPR 2017 | Runs the network on key frames and propagates features to other frames with optical flow. | Whole-frame propagation from key frames, flow network trained jointly. We do not skip frames (a project decision). | memory |
| Shelhamer, Rakelly, Hoffman, Darrell. **Clockwork Convnets for Video Semantic Segmentation.** ECCV 2016 Workshops | Updates deep layers less often than shallow ones, on a fixed or adaptive clock. | Layer-wise reuse in time; needs access inside the network. We treat the model as a black box. | abstract, 2026-11-16 |
| Li, Shi, Lin. **Low-Latency Video Semantic Segmentation.** arXiv 1804.00389 (CVPR 2018, venue from memory) | Feature propagation with spatially variant convolution and an adaptive key-frame scheduler based on predicted accuracy. | Trained propagation and scheduler; latency is their stated target, so their accounting is worth reading. | abstract (venue: memory), 2026-11-16 |
| Jain, Gonzalez. **Fast Semantic Segmentation on Video Using Block Motion-Based Feature Interpolation.** ECCV 2018 Workshops. arXiv 1803.07742 | Uses the block motion vectors of compressed video to propagate features between key frames. | Motion that costs nothing to compute, which is the cost that sank our Method 3 (alignment did not pay for itself). A direct answer to "why not use the codec's motion". | abstract, 2026-11-16 |
| Jain, Wang, Gonzalez. **Accel: A Corrective Fusion Network for Efficient Semantic Segmentation on Video.** CVPR 2019. arXiv 1807.06667 | Reference branch warps key-frame features with flow; a cheap update branch corrects them on each frame. | Trained two-branch system. Closest to "cheap pass every frame + memory", which is the shape of our Methods 4 and 5. | abstract, 2026-11-16 |
| Habibian, Abati, Cohen, Bejnordi. **Skip-Convolutions for Efficient Video Processing.** CVPR 2021. arXiv 2104.11487 | Convolutions computed only on the residual between frames, with learned gates. | Works inside the layers; reports computation (operations) saved. Our finding that FLOPs saved are not latency saved on a GPU applies. Author list from memory. | abstract, 2026-11-16 |
| Parger, Tang, Twigg, Keskin, Wang, Steinberger. **DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos.** CVPR 2022 | Sparse convolutions on frame differences with custom CUDA kernels, to turn the sparsity into real GPU speed. | The serious answer to "sparsity does not speed up a GPU": they needed custom kernels. We stay with stock kernels and a black-box model. Author list from memory. | abstract, 2026-11-16 |
| Hu, Caba, Wang, Lin, Sclaroff, Perazzi. **Temporally Distributed Networks for Fast Video Semantic Segmentation (TDNet).** CVPR 2020 | Splits a deep network into shallow sub-networks run on consecutive frames, recombined with attention. Cityscapes, CamVid, NYUD-v2. | Trained architecture. A different way to spend less per frame with memory; not a harness over a frozen model. | abstract, 2026-11-16 |

## Models and data we build on

| Paper | Role here | Checked |
|---|---|---|
| Xu, Xiong, Bhattacharyya. **PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers.** CVPR 2023 | Frozen backbone (PIDNet-S). | memory |
| Yu, Gao, Wang, Yu, Shen, Sang. **BiSeNet V2: Bilateral Network with Guided Aggregation for Real-Time Semantic Segmentation.** IJCV 2021 | Frozen backbone. | memory |
| Kim, Woo, Lee, Kweon. **Video Panoptic Segmentation.** CVPR 2020. arXiv 2006.11339 | Cityscapes-VPS, the video labels (6 labelled frames per 30-frame clip, instance ids kept across frames). | abstract, 2026-11-16 |
| Cordts et al. **The Cityscapes Dataset for Semantic Urban Scene Understanding.** CVPR 2016 | Source images and classes. | memory |

## To search next

- Video segmentation with an explicit persistent memory or canvas (for Method 4), beyond Patchwork.
- Foveated or log-polar sampling for segmentation, and any work testing warps on a **frozen** network (the direct competitor of Method 5's framing).
- Studies that compare efficiency methods against the same model at lower resolution at equal measured latency (support for our evaluation rule).
- Per-object or motion-aware recall metrics for driving scenes (for the moving-object metric).
