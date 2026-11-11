# Glass-Box Model Triad & Evaluation Architecture (`cvproj-mri`)

This document defines the mathematical, architectural, and tensor-level specifications for the three anomaly detection families in `cvproj-mri`, along with the evaluation and calibration contract required by [`eval/gate_rules.json`](eval/gate_rules.json).

---

## 1. Scientific Thesis & Problem Formulation

We study **unsupervised brain MRI anomaly detection (UAD) under adult-to-infant domain shift**:
- **Normative Adult Pretraining (`IXI`):** $N_{\text{train}} = 327$ healthy MNI152-aligned T1w volumes (`1 channel`).
- **Infant Developmental Adaptation (`BOBs`):** $N_{\text{train}} = 41$ infant subjects (`T1w` and paired `T1w + T2w`), where ongoing myelination inverts white/gray-matter T1/T2 contrast relative to adult scans.
- **Lesion Evaluation (`ATLAS v2.0` + Synthetic/Clinical Infant Lesions):**
  - **In-domain structural lesions:** `ATLAS v2.0` MNI152 T1w stroke scans with voxel-level lesion masks $Y \in \{0, 1\}^{D \times H \times W}$, stratified by lesion volume into small ($< 5\text{ mL}$, `auroc/lesion_small`), medium, and large strokes.
  - **Asymmetric vs. Bilateral Symmetric Stress-Test:** Unilateral focal anomalies vs. bilateral symmetric anomalies (e.g., hypoxic-ischemic basal ganglia/thalamus injury patterns in `BONBID-HIE` and controlled bilateral counterfactuals on held-out `BOBs` test scans).

### Core Hypothesis
1. **Standard reconstruction models (`Model 1: ResVAE`)** suffer high false-positive rates around normal cortical sulci/ventricles and degrade severely under adult-to-infant contrast shift.
2. **Pretrained vision foundation features (`Model 2: FoundationAD`)** improve semantic invariance over raw pixel residuals, allowing a direct comparison between **2.5D tri-slice Vision Foundation Models (`DINOv2` / `MedSAM`)** and **native 3D volumetric encoders (`3D SwinUNETR`)**.
3. **Bilateral hemispheric symmetry (`Model 3: SymmetryAD`)** acts as a **self-normalizing anatomical prior** across the mid-sagittal plane: left and right hemispheres myelinate synchronously in healthy infants, making contralateral feature comparison invariant to global age/scanner contrast shifts on asymmetric lesions—while requiring a gated normative branch so bilateral symmetric lesions are not subtracted away.

---

## 2. Input Tensor Conventions & Mid-Sagittal Alignment

All volumes are loaded in canonical **RAS+** orientation (`nib.as_closest_canonical`), foreground-masked ($X > 0$), and normalized per channel to $[0, 1]$ via 1st–99th foreground percentiles ([`transforms.py`](src/cvproj/data/transforms.py)).

| Representation Mode | Input Tensor Shape | Target Models | Context Mechanism |
| :--- | :--- | :--- | :--- |
| **2D Axial Baseline** | $(B, C, H, W)$ | Legacy [`Autoencoder`](src/cvproj/model.py) | Single slice $z$ |
| **2.5D Tri-Slice Stack** | $(B, C \times 3, H, W)$ | `ResVAE25D`, `FoundationAD` (`DINOv2`/`MedSAM`), `SymmetryAD` | Adjacent slices $[z - \Delta z, z, z + \Delta z]$ |
| **3D Volumetric / Sub-Volume** | $(B, C, D, H, W)$ | `ResVAE3D`, `FoundationAD` (`SwinUNETR`), `SymmetryAD3D` | Full 3D receptive field |

* **Mid-Sagittal Plane Convention (RAS+):** Along the sagittal $X$ (width $W$) axis, the anatomical midline lies at $x = W/2$ in MNI152 space (`IXI`, `ATLAS v2.0`) and centered RAS space (`BOBs`). Splitting a tensor $X \in \mathbb{R}^{B \times C \times \dots \times H \times W}$ along the last dimension yields:
  $$X_L = X[\dots, :, 0 : W/2], \qquad \tilde{X}_R = \text{flip}_W(X[\dots, :, W/2 : W])$$
  so that homologous left/right anatomical structures share identical spatial coordinates $(z, y, x)$.

---

## 3. Model Triad Specifications

### Model 0 (Legacy Smoke Baseline): 4-Layer Conv2D [`Autoencoder`](src/cvproj/model.py)
* **Architecture:** `Conv2d(C, W, 3, s=2) -> ReLU -> Conv2d(W, 2W, 3, s=2) -> ReLU -> ConvTranspose2d(2W, W, 4, s=2) -> ReLU -> ConvTranspose2d(W, C, 4, s=2) -> Sigmoid`.
* **Limitations (Why it fails in real UAD):**
  1. Shallow $4\times$ spatial compression with no bottleneck regularization allows near-identity copying of hyper/hypo-intense lesions.
  2. `ConvTranspose2d` introduces checkerboard artifacts at high-contrast ventricular boundaries.
  3. Pixel-wise MSE assumes isotropic Gaussian residuals, blurring sulcal boundaries and inflating false positives on healthy cortex.

---

### Model 1: 2.5D / 3D Residual Variational Autoencoder (`ResVAE`)

#### Architectural Design (Glass-Box PyTorch Implementation)
* **Normalization:** `GroupNorm(num_groups=8, num_channels=C)` (or `InstanceNorm`), **never `BatchNorm`**, because:
  1. 3D volumetric batches have small $B$ ($B \in [2, 8]$), causing noisy batch statistics.
  2. Running mean/variance in `BatchNorm` locks in adult `IXI` intensity statistics and breaks under infant `BOBs` contrast shift and batch-size-1 C++ inference.
* **Activation:** `SiLU` ($\text{SiLU}(x) = x \cdot \sigma(x)$) for smooth second-order gradients.
* **Encoder ($4$ stages, downsample factor $16\times$):**
  * Stem: `Conv(C_in, W, k=3, p=1)`
  * Stages $s \in \{1, 2, 3, 4\}$: Pre-activation Residual Block (`GroupNorm -> SiLU -> Conv(k=3) -> GroupNorm -> SiLU -> Conv(k=3) + residual`), followed by strided convolution downsampling (`k=3, s=2, p=1`).
  * Spatial Bottleneck: Maps feature map of shape $(B, 8W, H/16, W/16)$ (or $\times D/16$) via $1\times 1$ convolutions to variational parameters $\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2 \in \mathbb{R}^{B \times Z \times (H/16) \times (W/16)}$.
* **Reparameterization:**
  $$\mathbf{z} = \boldsymbol{\mu} + \exp\!\left(\tfrac{1}{2}\log\boldsymbol{\sigma}^2\right) \odot \boldsymbol{\epsilon}, \qquad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
  (Deterministic $\mathbf{z} = \boldsymbol{\mu}$ during inference and ONNX/C++ export).
* **Decoder (No Full-Resolution Skip Connections):**
  * **Crucial UAD Constraint:** Standard U-Net skip connections at $1/2$ and $1/4$ resolution pass lesion intensities directly around the bottleneck ("identity shortcut"). `ResVAE` uses a **pure bottleneck decoder** with **Bilinear/Trilinear Upsampling + `Conv(k=3, p=1)`** (eliminating `ConvTranspose` checkerboard artifacts) + Residual Blocks, ending in `Sigmoid`.

#### Training Objective
$$\mathcal{L}_{\text{ResVAE}}(X, \hat{X}) = \lambda_{L1} \frac{\|(X - \hat{X}) \odot M\|_1}{\|M\|_1} + \lambda_{\text{SSIM}}\bigl(1 - \text{SSIM}(X \odot M, \hat{X} \odot M)\bigr) + \beta(t) \, D_{\text{KL}}\!\left(q_\phi(\mathbf{z}|X) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\right)$$
where $M = \mathbb{I}[X > 0]$ is the foreground brain mask and $\beta(t)$ linearly warms up over the first $20\%$ of epochs to prevent posterior collapse.

---

### Model 2: Vision Foundation Feature Anomaly Detector (`FoundationAD`)

#### Architectural Design (Hybrid 2.5D vs. 3D Comparison)
Tests whether semantic patch representations from frozen pretrained backbones beat pixel-level generative reconstruction:
1. **2.5D Vision Foundation Branch (`DINOv2-ViT-B/14` or `MedSAM-ViT-B/16`):**
   * Takes a 3-slice stack $[z - \Delta z, z, z + \Delta z]$ resized to $(B, 3, 224, 224)$.
   * Extracts intermediate multi-layer patch token embeddings $F^{(l)} \in \mathbb{R}^{B \times N_p \times D_{\text{emb}}}$ (e.g., layers $l \in \{6, 9, 12\}$ concatenated and layer-normalized).
2. **3D Volumetric Medical Branch (`3D SwinUNETR` Encoder):**
   * Processes native 3D brain sub-volumes $(B, C, 96, 96, 96)$ via 3D windowed and shifted-window multi-head self-attention (`W-MSA` / `SW-MSA`), extracting hierarchical 3D feature maps.
3. **Normative Feature Scoring Head (Two Variants):**
   * **Parametric Gaussian / Mahalanobis Head:** Fits spatial patch-wise mean $\boldsymbol{\mu}_p$ and shrinkage covariance $\boldsymbol{\Sigma}_p + \lambda \mathbf{I}$ on healthy training features:
     $$A_{\text{Maha}}(p) = \sqrt{(f_p - \boldsymbol{\mu}_p)^\top (\boldsymbol{\Sigma}_p + \lambda \mathbf{I})^{-1} (f_p - \boldsymbol{\mu}_p)}$$
   * **Lightweight Feature Autoencoder (`FeatAE`):** A 3-layer $1\times 1$ bottleneck MLP/Conv head trained to reconstruct frozen foundation features $f_p \to \hat{f}_p$, scoring anomalies via cosine + $L_2$ feature residual:
     $$A_{\text{feat}}(p) = 1 - \frac{\langle f_p, \hat{f}_p \rangle}{\|f_p\|_2 \|\hat{f}_p\|_2} + \alpha \|f_p - \hat{f}_p\|_2^2$$
     upsampled back to native voxel resolution.

---

### Model 3: Bilateral Hemispheric Cross-Attention & Normative Network (`SymmetryAD`)

#### Architectural Design (Core Research Contribution)
Combines a **contrast-invariant contralateral symmetry branch** with a **normative reconstruction branch** in a single deterministic forward pass:

1. **Hemispheric Decomposition & Shared Multi-Scale Encoder:**
   * Split input $X$ into left hemisphere $X_L$ and flipped right hemisphere $\tilde{X}_R$.
   * Pass both through a shared residual encoder $E_\theta$ to obtain multi-scale feature maps $F_L^{(s)}, F_R^{(s)} \in \mathbb{R}^{B \times C_s \times H_s \times (W_s/2)}$ at scales $s \in \{1/4, 1/8\}$.
2. **Bilateral Hemispheric Cross-Attention Module (`BilateralCrossAttn`):**
   * At each scale $s$, query each spatial location in hemisphere $L$ against a local homologous neighborhood window $\mathcal{N}_r(u)$ in contralateral hemisphere $R$ (tolerating minor anatomical asymmetries up to radius $r$ voxels):
     $$Q_L = W_Q F_L^{(s)}, \quad K_R = W_K F_R^{(s)}, \quad V_R = W_V F_R^{(s)}$$
     $$\text{Contra}_{L \leftarrow R}(u) = \sum_{v \in \mathcal{N}_r(u)} \text{softmax}_v\!\left(\frac{\langle Q_L(u), K_R(v) \rangle}{\sqrt{d_k}} + B_{\text{rel}}(u - v)\right) V_R(v)$$
   * Symmetrically compute $\text{Contra}_{R \leftarrow L}$ by swapping $(L, R)$.
3. **Contralateral Asymmetry Residual & Gated Normative Fusion:**
   * Compute the structural asymmetry feature distance at each homologous coordinate:
     $$D_{\text{sym}}^{(s)}(u) = \left\| \text{LN}(F_L^{(s)}(u)) - \text{LN}(\text{Contra}_{L \leftarrow R}(u)) \right\|_2$$
   * Simultaneously decode a normative reconstruction $\hat{X}_L, \hat{X}_R$ from the bottleneck latent $\mathbf{z}$ (which does **not** subtract the opposite hemisphere, preserving sensitivity to bilateral symmetric lesions).
   * **Combined Voxel Anomaly Score $S(u)$:**
     $$S(u) = w_{\text{norm}} \, A_{\text{norm}}(u) + w_{\text{sym}} \, \text{Upsample}\!\left(D_{\text{sym}}(u)\right)$$
     where $(w_{\text{norm}}, w_{\text{sym}})$ is ablated across:
     - Pure Normative ($w_{\text{sym}} = 0$)
     - Pure Symmetry ($w_{\text{norm}} = 0$)
     - Hybrid Gated Fusion ($w_{\text{norm}}, w_{\text{sym}} > 0$, calibrated on validation data).

---

## 4. Anomaly Scoring, Calibration & Gate Metrics ([`eval/gate_rules.json`](eval/gate_rules.json))

To transition from healthy-only `reconstruction_mse` ([`eval/reconstruction_rules.json`](eval/reconstruction_rules.json)) to full clinical anomaly evaluation ([`eval/gate_rules.json`](eval/gate_rules.json)), [`evaluate.py`](src/cvproj/evaluate.py) computes:

1. **Voxel-Level Anomaly Map $S \in \mathbb{R}_{\ge 0}^{D \times H \times W}$:**
   * Masked to eroded brain foreground $M_{\text{brain}}$ to suppress skull-stripping boundary artifacts, optionally smoothed with a small 3D Gaussian kernel ($\sigma = 1.0\text{ voxel}$).
2. **Subject-Level Anomaly Score $s_i \in \mathbb{R}$:**
   * 99th-percentile (or top-$k\%$ mean) of $S(u)$ over $u \in M_{\text{brain}}$ for subject $i$.
3. **Probabilistic Calibration & Expected Calibration Error (`ece`):**
   * Fit a monotonic logistic calibration map $p_i = \sigma(a \cdot s_i + b)$ (Platt scaling) on validation split scores so anomaly scores become calibrated probabilities $p_i \in [0, 1]$ without test-split leakage.
   * Compute `ece` across $M = 10$ probability bins:
     $$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
4. **Gate Metrics Contract ([`eval/gate_rules.json`](eval/gate_rules.json)):**
   * `auroc`: Subject-level / slice-level AUROC (healthy vs. lesion-positive), `higher_is_better`, tolerance `0.01`.
   * `auprc`: Voxel-level / slice-level Precision-Recall AUC, `higher_is_better`, tolerance `0.01`.
   * `auroc/lesion_small`: Subgroup AUROC restricted to scans/slices with small lesion burden (bottom tertile / $< 5\text{ mL}$), `higher_is_better`, tolerance `0.02`.
   * `ece`: Expected Calibration Error, `lower_is_better`, tolerance `0.005`.
   * `latency_ms_p95`: 95th-percentile synchronized forward-pass latency per volume/batch on target device, `lower_is_better`, tolerance `2.0 ms`.
