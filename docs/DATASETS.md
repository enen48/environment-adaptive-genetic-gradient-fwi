# 数据来源、训练用途与存储

本项目的 AutoEncoder 学习 **velocity model → latent → velocity model**。
`model*.npy` 才是离线 AE 训练输入；`data*.npy` 是地震炮集，不能直接送给
velocity Encoder。原始 OpenFWI InversionNet 的监督任务与这里的 AE 任务不同。

## 仓库已经包含的数据

| 文件 | 样本来源（零基索引） | shape | 大小 | 实测速度范围 |
| --- | --- | --- | --- | --- |
| `data/sample/openfwi_flatvel_a_train.npy` | `model1.npy[0:64]` | `(64,1,70,70)` | 1,254,528 B | 1507–4500 m/s |
| `data/sample/openfwi_flatvel_a_val.npy` | `model49.npy[0:8]` | `(8,1,70,70)` | 156,928 B | 1502–4436 m/s |

两份都是实测验证过的 `float32` 原始速度模型子集；未修改数值。训练与验证
来自不同原始分片，没有把同一模型复制进两个集合。完整分片各有 500 个模型，
每份 9,800,128 字节，已经缓存在
`E:\Codex\environment-adaptive-genetic-gradient-fwi\data\openfwi`。
仓库只保存这 72 个小样本和来源清单。

原始分片 SHA256：

```text
model1.npy   6c721820e0b351315c9d0788bb1b085d6dc3820fe4e3dd73399b1496e35c1015
model49.npy  93d49930a714d6cc5af802c478803864ed9129d63251ca03a4b1093cd06ba1f5
```

下载采用 [ashynf/OpenFWI 非官方镜像](https://huggingface.co/datasets/ashynf/OpenFWI)，
固定版本 `9ed98af51d841bc7b7a43911021f87df6a730db5`，已对照镜像页面核验哈希。
精确下载 URL、索引和子集 SHA256 见
[`data/sample/provenance.json`](../data/sample/provenance.json)。尚未与官方
Google Drive 文件独立逐字节比对。

这是可复现的入门/冒烟训练集，不足以证明科研性能。`model49` 属于官方留出集；
这里称为 `val` 仅供烟雾测试。发表实验应从训练分片内另外划分调参集，保留
官方测试分片作最终一次评估。不要用测试模型初始化 Z-bank 或拟合 AE。

## 推荐扩展顺序

| 数据 | 项目中的建议用途 | 官方全量配对数据 | 官方训练/留出样本数 |
| --- | --- | --- | --- |
| FlatVel-A | AE 和反演闭环入门 | 43 GB | 24,000 / 6,000 |
| CurveVel-A/B | 曲层与分布外泛化 | 每种 43 GB | 每种 24,000 / 6,000 |
| FlatFault-A/B、CurveFault-A/B | 断层泛化与 QC 消融 | 每种 77 GB | 每种 48,000 / 6,000 |
| AGL Marmousi2 的 Vp | 独立复杂模型检验 | 大型独立包，本次未下载 | 不是独立模型训练样本集 |

OpenFWI 原始每模型 shape 为 `(1,70,70)`，每炮集为 `(5,1000,70)`。
Vel 家族训练为分片 1–48、留出为 49–60；Fault 家族训练为 1–96、留出为
97–108。来源：[官方数据页](https://openfwi-lanl.github.io/docs/data.html)、
[官方读取及划分说明](https://github.com/lanl/OpenFWI/blob/main/README.md)。

单个 float32 波形分片的数组载荷约为 700 MB：
`500 × 5 × 1000 × 70 × 4`。AE 阶段只下载速度模型即可；不要为了训练 AE
拉取几十 GB 的波形。完整 FlatVel-A 速度模型载荷约 588 MB，远小于配对全集。

官方下载入口：

- [FlatVel-A](https://drive.google.com/drive/folders/1NIdjiYhjWSV9NHn7ZEFYTpJxzvzxqYRb)
- [CurveVel-A](https://drive.google.com/drive/folders/1NGXnVG0gUFHfDcUvJxfozCgiI4WwquVk)
- [FlatFault-A](https://drive.google.com/drive/folders/1jOB6R_zewuFj5wZam7nDP7GixQnbnRLR)
- [CurveFault-A](https://drive.google.com/drive/folders/1vqUHJ-iRwp3ozL-e4HhKGpdO0e7NQZE1)
- 其余版本见[官方目录](https://openfwi-lanl.github.io/docs/data.html)。

## 数据许可证

OpenFWI 数据采用 **CC BY-NC-SA 4.0**。本仓库小样本保留署名、许可证链接、
来源和修改说明，见 [DATA_LICENSE.md](../data/sample/DATA_LICENSE.md)。
本项目代码的许可证与第三方数据许可证分开适用。

Marmousi2 有独立许可：[SEG AGL Elastic Marmousi 页面](https://wiki.seg.org/wiki/AGL_Elastic_Marmousi)
声明该版本采用 **CC BY 4.0**，版权归 2004 Allied Geophysical Laboratory,
University of Houston。应保留其原始版权声明、许可证与作者署名。官方列出的
[数据包](https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz)
本次 HTTP HEAD 返回 403，未下载、未声称核验其文件内容或大小。

建议仅从 Marmousi2 的 P 波速度模型裁剪/降采样并重新生成本项目的 **acoustic**
观测；它的原始 elastic 多分量数据不能被当成本项目简化声学求解器的等价观测。
保留独立空间区域，避免相邻重叠 patch 同时进入训练与测试。

## 官方采集参数与本项目 smoke solver 的区别

OpenFWI 论文表 3 对 Vel/Fault/Style 给出的网格间隔为 10 m，区域约
0.7 × 0.7 km，炮间距 140 m，接收点间距 10 m，时间采样 0.001 s，
记录约 1 s。[论文表 3](https://papers.nips.cc/paper/2022/file/27d3ef263c7cb8d542c4f9815a49b69b-Paper-Datasets_and_Benchmarks.pdf)

补充材料描述 2 阶时间 / 4 阶空间差分、15 Hz Ricker 子波、120 格吸收边界，
示例生成参数 `nt=1001`，而发布数据时间维为 1000。实际对接波形时必须查验
其取样/裁剪约定、炮检位置、子波振幅及时间零点，不能直接把表格值作为
已经验证的逐样本兼容配置。
[官方补充材料，第 3 节](https://proceedings.neurips.cc/paper_files/paper/2022/file/27d3ef263c7cb8d542c4f9815a49b69b-Supplemental-Datasets_and_Benchmarks.pdf)

本仓库的小型求解器仅用于方法闭环验证。使用它重新正演已下载速度模型时，
应将结果明确标记为“本项目重新生成的 acoustic synthetic observations”，
不能声称重现了 OpenFWI 的原始炮集。速度归一化需覆盖 1500–4500 m/s，
不应静默截断为合成演示的较小范围。

## GitHub 与 E 盘

普通 Git 只跟踪代码、配置、来源清单以及上述小数据。GitHub 网页上传每文件
上限 25 MiB，普通 Git 文件硬上限 100 MiB；大型波形分片不直接提交。
进阶数据在 E 盘缓存，以固定 URL、SHA256 和下载脚本复现，必要时单独配置
Git LFS 或数据发行附件。[GitHub 官方限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)

下载、解压、训练缓存优先指向 `E:\Codex\environment-adaptive-genetic-gradient-fwi`；
数据加载使用 `numpy.load(..., mmap_mode="r", allow_pickle=False)`，按 batch
读取。E 盘是磁盘存储，无法增加物理 RAM 或 GPU 显存。

