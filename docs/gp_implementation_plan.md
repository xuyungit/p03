# 蓝图

直接落地的蓝图，实现 Kriging/GP 的系统方案讲清楚，便下面按「目标—数据—模型—训练—评估—不确定性—对比实验—扩展」八个板块给出步骤与决策点。参考现有的bridge_nn.py的实现。实现灵活的表格类数据的训练。模型的输入输出可以灵活定义，使用正则表达式的方式指定从训练数据的哪些column用作输入，哪些用作输出。复用模型的评价指标计算和评价可视化。

0. 目标与约束

目标
	•	用高斯过程家族（Kriging/GP）构建你的替代模型（surrogate），在精度与不确定性估计上与当前 NN 做对比。
	•	你的数据：N=8000，输入 dₓ=19，输出 dᵧ=71。

约束
	•	经典 GP 训练复杂度 O(N³)；N=8000 时不可直接用标准Kriging/GP。
	•	需要：稀疏/变分近似 + 多任务处理 + 输出降维 才能“既能跑又好用”。

1. 数据与预处理
	1.	数据划分

	•	训练/验证 = 从当前训练数据csv中85/15划分。保证样本分布一致，防止泄漏。

	2.	标准化

	•	输入 X：z-score（按训练集均值/方差）；
	•	输出 Y：z-score（每一维独立标准化，便于多任务权重均衡）。

	3.	输出降维（强烈推荐）

	•	对 71 维输出做 PCA（或 PLS 回归型降维），保留 95–98% 方差，得到 k（通常 8–15）。
	•	后续只拟合 k 个主成分，推理后再 逆PCA 重构回 71 维。
	•	好处：显著降训练与内存负担；同时让多任务相关性变“可压缩”。


2. 模型族与架构选择（从易到难的三档）

A 档（首选基线）：PCA + k 个独立的稀疏变分 GP (SVGP)
	•	对每个主成分 yⱼ 训练一个 SVGP；核函数用 RBF/Matérn + ARD（自动相关性，能做输入选择）；
	•	每个 SVGP 选诱导点 m∈[300, 800]；小批量训练（mini-batch 对 N=8000 有意义）；
	•	简单稳健、易调参；是与 NN 对比的首个可靠基线。

B 档（更强的多任务）：PCA + 多任务 GP（ICM/LMC 核）
	•	建一个多输出 GP，共享输入核（ARD）+ 学习任务协方差（捕捉输出相关性）；
	•	适合 PC 之间仍显著相关的情况；计算更复杂，但样本效率高。
	•	诱导点与 A 档相当；需要成熟实现（如 GPyTorch/GPflow 的多任务变分）。

C 档（非线性更强）：DKL（Deep Kernel Learning）
	•	前端用一个小 NN 做特征映射 φ(X)，后端接 GP（可仍是 SVGP/多任务）；
	•	既保留 GP 的不确定性，又提升对复杂非线性的表达；
	•	训练更难，需要分段学习率/先冻后放策略与更强正则。

建议路线：先 A 档打通 → 若全局指标接近或优于 NN，再尝试 B 档；若遇到强非线性再上 C 档。

3. 关键超参与默认值（给你开箱即用的“起跑线”）
	•	核函数：RBF（先用）或 Matérn-5/2（更鲁棒）；都用 ARD（每个输入一个长度尺度）。
	•	诱导点 m：500（起步），做灵敏度试验 m=300/500/800。
	•	优化目标：变分 ELBO / 负对数似然（NLL）最小化。
	•	优化器：Adam（1e-2 → 1e-3 余弦退火），核超参可配合自然梯度/分组学习率。
	•	批大小：256–1024（按显存/内存调）。
	•	早停：验证集 NLL 或 RMSE 不再下降（耐心 10–20 个 epoch）。
	•	稳定化：核长度尺度、噪声方差用对数域参数化；对角抖动（jitter）1e-6~1e-4。

4. 训练流程（逐步可操作）
	1.	做 PCA：Y→(Z₁…Z_k)，记录均值/方差/投影矩阵，测试时复用。
	2.	建 A 档模型：为每个 Zⱼ 建 SVGP（相同核族、不同参数）；
	3.	诱导点初始化：对 X 做 k-means 取 m 个中心（更稳定），或从训练集中按密度采样；
	4.	训练：
	•	先只训练似然噪声+核长度尺度（其余冻结）若干步，让模型“收住”；
	•	然后全参联合训；
	•	记录验证集 NLL / RMSE；保存最佳 checkpoint。
	5.	推理：得到 Z 的均值与方差 → 逆 PCA 重构回 71 维的均值与协方差对角（近似）。
	6.	可视化：关键输出维（或物理关切的指标）画预测±置信带，对比 NN 的点估计。

5. 评估指标与不确定性校准

点估计
	•	每维 MAE/RMSE/R²；
	•	物理关键维度做单独报表（Top-K），其余做加权平均（按工程重要度或量纲归一化后平均）。

不确定性
	•	预测分布（均值±σ），统计 置信区间覆盖率：
	•	90% 置信区间应覆盖 ≈90% 的真实点（校准良好）；
	•	校准曲线（预测分位 vs 经验分位）与 NLL；
	•	必要时做 温度缩放/方差重标定（在验证集上优化一个缩放因子），提升校准度。

速度
	•	训练时长、单样本推理延迟（CPU/GPU）；
	•	记录显存峰值；为上线评估提供依据。

6. 与神经网络的公平对比清单
	•	同一数据划分与标准化；
	•	同一输出空间：建议 NN 也在 PCA 空间（k 维）回归 + 逆 PCA 重构，确保任务难度一致；
	•	统一指标：MAE/RMSE/R²/NLL/覆盖率；
	•	同等资源限制：如总训练时长或算力预算接近；
	•	统计显著性：多次重启（≥3 次），报告均值±标准差；对关键指标做配对 t 检验。

7. 灵敏度与可解释性（加分项）
	•	ARD 长度尺度：从 GP 的 ARD 参数中读取“每个输入维度的重要性”（长度尺度越小，越敏感）。
	•	全局敏感性（可选）：在 A 档 SVGP 的代理上做 Sobol 指数（对 PCA 主成分或对重构后的关键输出维），量化主/总效应；
	•	对比 NN 的输入重要性（如基于梯度/SHAP），两边交叉验证物理一致性。

8. 常见坑位与规避
	•	输出降维过激：PCA 方差阈值别太低（95–98%），否则逆映射误差影响结论。
	•	噪声项过小：会导致过拟合与数值不稳定；允许各任务（主成分）有独立噪声。
	•	诱导点太少：拟合欠佳；太多则训练慢且易不稳。先 500 做灵敏度。
	•	异方差：若不同区域噪声不同，考虑输入依赖噪声（更复杂）或分区建模。
	•	分布漂移：确保测试集与训练集分布一致；否则报告不确定性失配并给出外推警示。

9. 结论性决策树（你可以照此推进）
	1.	起步：PCA→k≈10；A 档（独立 SVGP, m=500, ARD）；完成一版完整评估。
	2.	若精度接近或优于 NN，且校准更好：
	•	升级到 B 档（ICM/LMC 多任务）；
	•	或固定 A 档做工程上线（NN 在线推断 + GP 做离线校准/复核）。
	3.	若低于 NN：
	•	尝试 DKL（小 NN + GP）；
	•	增大 m；
	•	检查输出 PCA 阈值与输入标准化/异常值。
	4.	交付：选择“性能/成本/校准”最平衡的一档为基线；另外保留 GP 输出的置信区间供决策使用。

10. 可扩展方向（按需选用）
	•	分组多任务：如果 71 维按物理可分组（如跨间、测点、量纲），每组建一个多任务 GP。
	•	层次核/分层 GP：先建结构级变量（温度、沉降）对输出的 GP，再对残差建局部 GP。
	•	物理先验：将已知的线性/单调关系编码到核函数（例如对某些维度使用有向核、周期核等）。
	•	主动学习：用 GP 的方差引导再采样（选择不确定性最大的点重跑 FEM），最少试验达最大收益。


——— 以下为更具体的“落地实施计划”与代码级细化 ——

11. 模块与文件结构（与现有代码风格一致）
- 新增目录与脚本（首期仅实现 A 档基线，可逐步扩展）：
  - `src/models/gp/common.py`：通用工具
    - 列选择正则解析与校验（复用/抽取 bridge_nn 与 tree baseline 的实现）
    - PCA/PLS 管道：fit/transform/inverse_transform，保存/加载
    - 训练/验证拆分、数据加载（PyTorch DataLoader，小批量）
    - 指标与可视化对接：调用 `models.evaluation.report.EvaluationReport`
  - `src/models/gp/svgp_baseline.py`：A 档 PCA + 独立 SVGP 训练脚本（主路径）
  - `src/models/gp/mt_icm.py`：B 档多任务变分 GP（ICM/LMC）（第二阶段）
  - `src/models/gp/dkl.py`：C 档 DKL（第三阶段）
  - `src/models/gp/infer.py`：推理封装（加载 scalers + PCA + GPs，输出均值/方差）
- 可选小型重构：将“列选择+校验”共用逻辑抽到 `src/models/utils/columns.py`（供 NN、Tree、GP 共用）

12. 依赖与环境 (已经完成安装)
- 在 `pyproject.toml` 中新增（按需安装，确保与 PyTorch 版本匹配）：
  - `gpytorch`（SVGP/多任务/核函数支持）
  - 可选：`botorch`（后续主动学习/高级内核初始化）
- GPU 可选；CPU 也可运行（批量和诱导点数需适当减小）。

13. CLI 设计（与 bridge_nn.py / tree baseline 对齐）
- 通用数据/列参数（完全复用现有风格）：
  - `--train-csv ... --test-csv ...`
  - `--input-cols` / `--target-cols` 或 `--input-cols-re` / `--target-cols-re`
  - `--val-ratio`, `--augment-flip/--no-augment-flip`, `--augment-profile`, `--augment-config`
  - `--preset path.json`（可覆盖数据与超参）
- PCA/降维相关：
  - `--pca-enable`（默认开）
  - `--pca-variance 0.97`（优先用方差阈值）或 `--pca-components k`
  - 可选 `--pls-components k`（替代 PCA；二选一）
- SVGP 相关（A 档）：
  - `--kernel {rbf,matern52}`，`--ard`（开关 ARD），`--noise-init 1e-3`，`--jitter 1e-6`
  - `--inducing m`（默认 500），`--inducing-init {kmeans,random}`
  - `--batch-size 512`，`--epochs 2000`，`--lr 1e-2`，`--lr-decay cosine|step|none`
  - `--whiten`（变分白化，默认开），`--natural-grad`（可选，对变分参数）
  - `--early-stopping --patience 50 --min-delta 0.0`
- 不确定性/推理：
  - `--save-std`（保存重构后每一维输出的近似标准差）
  - `--coverage-list 0.5,0.9,0.95`（评估覆盖率）
- 资源与复现实验：
  - `--seed`, `--experiment-root`, `--print-freq`, `--val-interval`

14. 训练/评估流程（代码级细化）
- 数据阶段：
  - 按 NN/Tree 相同逻辑解析列集（显式列表优先，其次正则）。
  - 训练 CSV 可多文件合并；测试 CSV 支持多文件并生成“合并测试集 + 单文件测试集”评估结果（保持一致）。
  - 训练集可选左右翻增强（与现有 `augment_config` 对齐）。
  - `StandardScaler` 分别拟合 X 与 Y（只基于训练集，含增强样本），并持久化保存。
  - 训练/验证切分：`train_test_split`（按 `--val-ratio`）。
- PCA/PLS：
  - 在已标准化的 `Y_s` 上拟合 PCA（或 PLS），优先以 `--pca-variance` 确定 `k`；保存 `mean_ / components_ / explained_variance_ratio_` 等。
  - 训练时将 `Y_s` 投影为 `Z`（k 维），GP 只拟合 `Z`。
  - 推理后将 `Z_pred`（及其方差）逆映射回 `Y_s`，再逆标准化至原单位。
  - 协方差近似：以对角近似将 `Var[Y_s] ≈ diag(W^T Var[Z] W)`，仅保存每个输出维度的 `std`（可在 B 档/后续扩展成全协方差的低秩近似）。
- A 档（SVGP，独立 k 个任务）：
  - 为每个主成分 `Z_j` 构建一个单输出 SVGP：核 `RBF`（默认）或 `Matern-5/2` + `ARD`；诱导点 m=500；`whiten=True`。
  - 诱导点初始化：`kmeans`（默认，基于 X_s）；备选 `random`。
  - 训练循环（小批量）：
    - 可选预热阶段：仅更新噪声方差与长度尺度若干 epoch；随后解冻全参联合训练。
    - 优化：Adam；学习率调度（cosine/step/none）；可选对变分参数用自然梯度（迭代交替）。
    - 验证：每 `val_interval` 个 epoch 在验证集计算 NLL/RMSE；保存 best/last。
  - 评估：
    - 生成 `train/val/test` 与每个单独测试文件的预测 CSV（均值），以及 `*_std.csv`（若 `--save-std`）。
    - 通过 `EvaluationReport` 产出 `*_metrics.json`、`*_predictions.png`、`*_residuals.png`、文本报告等（与 NN/Tree 一致）。
    - 计算并保存不确定性覆盖率、NLL、校准曲线（见第 15 节）。
- 产物保存：
  - 目录：`experiments/gp_svgp/<timestamp>/`
  - `config.json`（CLI + 解析后列名 + PCA 信息 + SVGP 超参）
  - `checkpoints/`：每个主成分一个子目录，保存 gpytorch state_dict、诱导点、核超参；另存 `scaler_x.pkl`、`scaler_y.pkl`、`pca.pkl`
  - `train_* / val_* / test_*` 指标与图表，与现有脚本命名一致，便于横向比较。

15. 不确定性度量与校准（实现要点）
- 预测：获取每个 `Z_j` 的预测均值 `μ_j` 与方差 `σ_j^2`；堆叠为 `μ_Z, Var_Z=diag(σ^2)`。
- 逆映射：`μ_Ys = μ_Z W`；`std_Ys ≈ sqrt(diag(W^T Var_Z W))`；再通过 `scaler_y` 变回原单位（std 要乘以 `scaler_y.scale_`）。
- 覆盖率：对每个样本/维度，统计 `[μ ± z_α σ]` 是否覆盖真实值；默认 α=1.64(90%)、1.96(95%)；输出总体覆盖率与分维覆盖率。
- NLL：以高斯似然近似 `-log p(y|μ,σ)` 的均值（原单位）。
- 温度缩放：在验证集搜索一个全局方差缩放系数 `τ`，使目标覆盖率更贴近名义值（保存 `tau.json`）。

16. 与现有代码的对齐与复用
- 列选择/校验：直接复用 `bridge_nn.py`/`rev03_rtd_nf_e3_tree.py` 的正则解析与“逐文件校验”逻辑，抽取为公共函数，避免多处重复。
- 数据增强：与现有 `augment_config`/`load_augmentor` 对齐；flip 规则一致。
- 指标与可视化：统一使用 `models.evaluation.report.EvaluationReport`，从而产物与 NN/Tree 完全一致，便于对比。
- 运行目录结构、`config.json` 字段命名、文本/图表产物命名，对齐现有脚本，减少学习成本。

17. 对比实验规范（可直接执行）
- 统一数据/列配置、增强与标准化方式；NN 建议也在 PCA 空间回归后逆映射，确保任务难度一致。
- 统一指标：MAE/RMSE/R²/NLL/覆盖率；每个方案至少 3 次不同 seed 重启，报告均值±标准差。
- 统一资源约束：限定训练总时长或 GPU 占用，确保公平。
- 示例命令：
  - A 档（SVGP 基线）
    - `uv run python src/models/gp/svgp_baseline.py --train-csv data/d03_all_train.csv --test-csv data/d03_all_test.csv --input-cols-re "^R|^N|^F" --target-cols-re "^D|^Ry|^A" --pca-variance 0.97 --kernel rbf --ard --inducing 500 --batch-size 1024 --epochs 2000 --early-stopping --patience 50`
  - Tree/NN（现有）对应命令不变，保持完全相同的列与数据划分。

18. 验收标准（Definition of Done）
- 代码：`svgp_baseline.py` 可单条命令完整训练并生成与 NN/Tree 等价的产物集；`config.json` 记录完整配置与列名；`checkpoints/` 含 GP 状态、scalers、PCA。
- 指标：在小规模试运行（如 N≈3k, k≈10, m≈500）下，RMSE/MAE 与 NN 同量级；覆盖率曲线合理（90% 区间覆盖率 ~90%±5%）。
- 复现：连续两次相同配置训练，产物（除时间戳）一致；加载 `infer.py` 并在任意 CSV 上得到均值/方差输出。
- 文档：本文件（计划）与 `docs/` 使用说明更新；给出至少 2 组可复现实验命令与对应 `experiments/...` 路径。

19. 测试计划（pytest）
- 单元测试：
  - 列选择与正则解析（空/非法正则/重复列去重/多文件一致性报错）。
  - PCA 管道：fit/transform/inverse_transform 误差与解释方差阈值满足预期。
  - 方差逆映射：构造已知投影的玩具例子，校验 `diag(W^T Var_Z W)` 实现正确。
- 集成/冒烟：
  - 人工小数据（N≤512, d_x≈5, d_y≈6）完整跑通 SVGP，产出 `*_metrics.json` 与图表且无异常。
  - 推理：`infer.py` 加载已训练产物，对输入 DataFrame 输出均值/方差，形状/列名匹配训练列。

20. 第二/三阶段扩展（B 档、C 档）
- B 档 ICM/LMC：
  - 以 PCA 后的 k 维为任务维，建 gpytorch 多任务变分 GP，学习任务协方差；
  - 诱导点策略与 A 档一致；
  - 对比：同等 m 与 epoch，下验证指标与覆盖率能否显著提升；
  - 产物：除均值/方差外，额外保存任务协方差矩阵与其可视化。
- C 档 DKL：
  - 小型前端 MLP（2–3 层、宽度 64–128）作特征映射 φ(X)，后接 SVGP；
  - 训练策略：先冻结 GP，仅训 φ，后联合微调；学习率分组（φ 较大、核/似然较小）。

21. 风险与回滚
- gpytorch 与当前 torch 版本的兼容性需确认；若受限，可先用 sklearn GaussianProcessRegressor 在“子采样 + k 小数据”上做功能验证，再切至 SVGP。
- 诱导点过大/批量过大导致显存溢出：预留 `--batch-size`、`--inducing`、`--whiten`、`--jitter` 等兜底参数；优先保证训练稳定。
- PCA 阈值过低造成逆映射误差：在 `config.json` 与评估报告中显式记录 `explained_variance_ratio_` 与重构误差统计，避免误读。

22. 里程碑与时间线（建议）
- T+0.5d：抽取列解析公用函数；搭好 `gp/common.py` 框架；编写 PCA 管道与保存/加载；本地小数据冒烟通过。
- T+1.5d：实现 `svgp_baseline.py`（A 档），完成训练循环、验证、产物写出；接入 `EvaluationReport`。
- T+2d：完成不确定性覆盖率/NLL/温度缩放；撰写使用文档；在真实数据（N≈8k, k≈10, m≈500）跑通一次。
- T+3d：根据效果选择推进 B 档或调优 A 档（m/核/学习率），并补齐测试用例。

23. TODO 清单（执行追踪）
- 文档与计划
  - [x] 完善 GP 实施蓝图与细化计划（本文件）

- 依赖与环境
  - [x] 在 `pyproject.toml` 添加 `gpytorch`（可选 `botorch`）
  - [x] 使用 `uv sync` 验证安装与版本兼容（与 PyTorch）

- 公共工具与数据管道
  - [x] 抽取列解析/校验为公共函数（`models/utils/columns.py`）
  - [x] 在 `models/gp/common.py` 实现：
    - [x] PCA/PLS 管道：fit/transform/inverse_transform，持久化保存/加载
    - [x] 数据加载与标准化（与 `TabularDataModule` 对齐的小批量接口）
    - [x] 与 `EvaluationReport` 的评估对接
  - [x] 接入/复用 `augment_config` 翻转增强

- A 档：SVGP 基线
  - [x] 实现 `src/models/gp/svgp_baseline.py`（CLI 与 NN/Tree 对齐）
  - [x] 支持核选择（RBF/Matérn）、ARD、诱导点初始化（kmeans/random）
  - [ ] 小批量训练循环（预热→联合），学习率调度与早停
  - [ ] 产物保存：`config.json`、`checkpoints`、`scaler_x.pkl`、`scaler_y.pkl`、`pca.pkl`、预测 CSV/图表
  - [ ] 不确定性：保存 `std`、覆盖率、NLL、温度缩放 `τ`
  - [ ] 推理：`src/models/gp/infer.py` 输出均值/方差
  - [ ] 冒烟：小规模数据完整跑通并生成产物

- 测试
  - [x] 单元：列解析（`resolve_from_csvs` 正则/空/异常）
  - [ ] 单元：PCA 逆映射精度
  - [x] 单元：方差投影（`diag(W^T Var_Z W)`）
  - [ ] 集成：`svgp_baseline.py` 全流程冒烟
  - [ ] 推理：`infer.py` 输入/输出列对齐检查

- 对比与报告
  - [ ] 与 NN/Tree 公平对比（统一列/增强/标准化）
  - [ ] 统计显著性（≥3 个随机种子）
  - [ ] 在 `docs/` 补充使用命令与结果示例

- B 档：ICM/LMC 多任务（第二阶段）
  - [ ] 实现 `src/models/gp/mt_icm.py`（变分多任务）
  - [ ] 任务协方差保存与可视化
  - [ ] 对比实验与结论

- C 档：DKL（第三阶段）
  - [ ] 实现 `src/models/gp/dkl.py`（小型 MLP + SVGP）
  - [ ] 分段训练策略与学习率分组
  - [ ] 对比实验与结论

- 风险缓解
  - [ ] 显存/内存溢出回退策略（m、batch、jitter、whiten）
  - [ ] 记录 PCA 解释方差与重构误差，监控降维偏差

- 里程碑检查
  - [x] T+0.5d 完成公共函数与 PCA 管道
  - [ ] T+1.5d SVGP 基线训练/评估打通
  - [ ] T+2d 不确定性指标与文档
  - [ ] T+3d 推进 B/C 或调优 A
