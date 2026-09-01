# resprpy

Python 移植版 **respR 2.3.4**（水生态呼吸代谢分析 R 包，Harianto & Carey,
*Methods in Ecology and Evolution* 2019, doi:10.1111/2041-210X.13162）。
目标是：**Python 数值输出与 R 完全一致**（用 R 4.5.3 实测生成参考 CSV，
pytest 逐值对拍）。许可 GPL-3。

## 安装

```bash
pip install -e .            # 或 pip install -e ".[plot,test]"
```

依赖：numpy、scipy（可选 matplotlib、pytest、openpyxl）。

## respR 与 resprpy 对比

### 功能覆盖（respR 2.3.4 全部 27 个导出函数）

| respR (R) | resprpy (Python) | 状态 | 对拍精度 |
|---|---|---|---|
| `calc_rate` | `calc_rate` | ✅ 已对拍 | slope/intercept/rsq ≤1e-13 |
| `calc_rate.int` | `calc_rate_int` | ✅ 已对拍 | ≤1e-6 |
| `calc_rate.bg` | `calc_rate_bg` | ✅ 已对拍 | ≤1e-6 |
| `calc_rate.ft` | `calc_rate_ft` | ✅ 已对拍（vec/df/insp/width） | ≤1e-5 |
| `auto_rate` | `auto_rate` | ✅ 已对拍 | bw.SJ/density 逐位一致（≤2e-13） |
| `auto_rate.int` | `auto_rate_int` | ✅ 已对拍 | ≤1e-5 |
| `adjust_rate` | `adjust_rate` | ✅ 已对拍（6 方法） | ≤1e-6 |
| `adjust_rate.ft` | `adjust_rate_ft` | ✅ 已对拍 | ≤1e-6 |
| `select_rate` | `select_rate` | ✅ 已对拍（34 方法全覆盖） | ≤1e-8 |
| `select_rate.ft` | `select_rate_ft` | ✅ 已对拍 | ≤1e-5 |
| `convert_DO` | `convert_DO` | ✅ 已对拍 | ≤1e-12 |
| `convert_MR` | `convert_MR` | ✅ 已对拍 | ≤1e-12 |
| `convert_rate` | `convert_rate` | ✅ 已对拍 | ≤1e-12 |
| `convert_rate.ft` | `convert_rate_ft` | ✅ 已对拍 | ≤1e-8 |
| `convert_val` | `convert_val` | ✅ 已对拍 | ≤1e-12 |
| `format_time` | `format_time` | ✅ 已对拍 | 秒差+1 逐值一致 |
| `inspect` | `inspect` | ✅ 已对拍 | checks/locs 一致 |
| `inspect.ft` | `inspect_ft` | ✅ 已对拍 | checks/dataframe 一致 |
| `oxy_crit` | `oxy_crit` | ✅ 已对拍（bsr + segmented） | bsr ≤1e-6；seg crit=7.719111 逐位一致 |
| `import_file` | `import_file` | ✅ 已对拍（NeoFox） | 最大相对误差 0.0 |
| `sim_data` | `sim_data` | ✅ 已对拍（结构） | 结构一致 |
| `subsample` | `subsample` | ✅ 已对拍 | — |
| `subset_data` | `subset_data` | ✅ 已对拍 | — |
| `select` | `select` | ✅ 等价 | — |
| `%>%` | （无；Python 直接链式调用） | ➖ 不适用 | — |
| `test_lin` | `test_lin` | ✅ 已对拍（RNG 不同，结构一致） | — |
| `unit_args` | `unit_args` | ✅ 已对拍 | — |

✅ = 与 R 逐值对拍通过（pytest 全绿）；🟡 = 代码已按 R 源码移植，缺真实样本文件对拍。

### 数值一致性（已验证部分）

- `calc_rate`：60 段真实数据回归（X:\Rtools\20260422 随机文件），
  intercept/slope/rsq/row/time/oxy 全部与 R 一致（最大相对差 ~1e-13）。
- `auto_rate`：滚动回归 218 行逐值一致；`bw.SJ` 带宽 0.0006344835 逐位一致；
  `density()` 512 网格点最大相对差 2.05e-13；10 个峰位置/数值全一致；
  linear 方法 summary 8 行与 R 完全一致。
- `convert_*`：单位换算 6 个物理锚点全中（sw_dens=1024.64077347883、
  molvol=24.03038425、mg/L→%Air=108.4419 等）。
- `format_time`：lubridate 风格时间解析，秒差+1 逐值一致。
- `select_rate`：34 个方法全覆盖对拍（R 侧 4 个方法因输入对象限制报错，
  Python 报错行为一致；30 个可运行方法 summary 数值 ≤1e-8）。
- `oxy_crit(segmented)`：crit=7.719111 与 R 逐位一致（含 R MT19937 随机
  序列复刻：`set.seed(1)` 的 runif/sample.int 逐位相同）；summary 6 列
  最大相对差 4.3e-12、seg_fit 245 行 4.5e-15。
- `import_file`：NeoFox 样本（ACACTB11.csv，6031×5）与 R 最大相对误差 0.0。
- 0_RMR.R 全流程（8 通道 × 36 循环）复刻对拍一致。

### 已知差异 / 限制

| 项目 | respR (R) | resprpy (Python) |
|---|---|---|
| 输入格式 | data.frame / data.table | numpy 2D 数组（两列：time, oxygen） |
| 输出对象 | S3 对象（`$rate`、`$summary`） | dataclass（`.rate`、`.summary` dict）或 dict |
| 函数命名 | 点号 `calc_rate.int`、`oxy.unit` | 下划线 `calc_rate_int`、`oxy_unit` |
| row/endrow | 1-based | **1-based（与 R 一致）** |
| `oxy_crit` segmented 法 | R `segmented` 包（Muggeo 2003，bootstrap-restart） | ✅ 已移植+对拍（含 R MT19937 RNG 逐位复刻） |
| 绘图 | S3 `plot()`（inspect/calc_rate/auto_rate/oxy_crit 等） | ✅ `plot_inspect/plot_calc_rate/plot_auto_rate/plot_oxy_crit`（matplotlib，配色布局对齐 R） |
| `test_lin` 随机数 | R 原生 RNG | 已移植 R MT19937（`set.seed` 逐位一致）；与 R 的 `sample` 序列相同 |
| `import_file` | 已弃用 | ✅ 已移植；NeoFox 样本逐值一致（Witrox/AutoResp 因 R 自身 fread 误解析不作基准） |
| `%>%` 管道 | magrittr | 无（Python 链式调用替代） |
| 依赖 | data.table/roll/marelac/segmented 等 20+ 包 | numpy/scipy（matplotlib 可选） |
| 性能 | 滚动回归等为 C 代码 | 纯 Python/numpy（同量级，窗口回归略慢） |
| 许可 | GPL-3 | GPL-3 |

### 移植架构

```
src/resprpy/
├── _marelac.py      # marelac 2.1.11 物理常数（vapor/gas_solubility/sw_dens/sw_gibbs...）
├── _units.py        # 单位匹配/换算/标签（84 个正则表自动生成）
├── _unit_regexes.py # 单位正则表（gen_code.py 从 R 源码自动提取）
├── _gibbs_coeffs.py # Gibbs 系数矩阵（自动提取）
├── convert.py       # convert_DO/MR/rate/val
├── convertft.py     # convert_rate.ft
├── calc.py          # calc_rate
├── intflow.py       # calc_rate.int/bg/ft, auto_rate.int
├── auto.py          # auto_rate（滚动回归+KDE+峰检测+zeroin2+bw_sj 全链）
├── adjust.py        # adjust_rate/ft
├── selectrate.py    # select_rate/ft, test_lin
├── oxycrit.py       # oxy_crit（bsr 断棒 + segmented Muggeo 完整移植含 boot）
├── _rng.py          # R MT19937（RNG_Init/MT_genrand/R_unif_index 逐位复刻）
├── plots.py         # plot_inspect/calc_rate/auto_rate/oxy_crit（matplotlib）
├── inspectmod.py    # inspect/ft（6 项 QC 检查）
├── importers.py     # import_file + 14 个仪器解析器
├── data.py          # sim_data/subsample/subset_data/select
└── timefmt.py       # format_time
```

关键复刻细节（与 R 逐位一致的根因）：
1. R `zeroin2`（`uniroot` 底层）的 C 语序赋值 `a=b; b=c; c=a`，Python 逐语句复刻；
2. R `fft(inverse=TRUE)` 不归一化（≠ numpy ifft÷N），需手工对齐；
3. respR 的 max/min 方法交叉调用怪癖（`method="max"` 实际跑 min 排序）；
4. R TRE 正则 `\b` 边界语义与 `(?i)` 内联标志翻译；
5. `density()` 的 ext=4 宽网格 + C_BinDist 线性分箱 + 2n FFT 互相关；
6. 单位匹配走命名正则对象（非枚举），完整复刻 R 的匹配机制；
7. R 的 `Int32` 是 **unsigned int**：MT19937 全部无符号算术（逻辑右移），
   `RNG_Init` 50 步 LCG 加扰 + 625 填充 + FixupSeeds；`R_unif_index` 用
   16 位块拼接 + 拒绝采样（R 4.3+ 默认）；`set.seed(1)` 序列逐位一致；
8. segmented 包 2.2-1 完整移植：Muggeo 迭代（`_brent_fmin` 优化步长、
   `far.psi`/`in.psi`/`adj.psi`/tabulate 从 bin 1 起算）+ bootstrap-restart
   主循环（seed 由 `mean(y)` 字符串去零派生，n.boot=10 重抽样 + 停滞检测 +
   psi 拉开逻辑）；
9. `generate_mrdf` 的滚动斜率用 `solve(X'X, X'y)`（= R `roll::roll_lm`
   增量算法），不用 QR lstsq；`broken_stick` 的 `dt[1:n]` 是 n 行、
   splitpoint 取 `x[n-1]`。

## 测试

```bash
python -m pytest tests/ -v
```

参考数据 `tests/reference/` 全部由 `reference/gen_refs_*.R` 在 R 4.5.3 +
respR 2.3.4 实测生成，pytest 逐元素相对误差对比（容差分级：
回归系数 1e-8、rsq 1e-12、row/endrow 1e-12、time 1e-9、density 1e-3）。
segmented 参考（`ref_oxy_crit_seg_*.csv`）由 `gen_refs_seg.R` 在 R +
segmented 2.2-1 下生成；`plot_*` 冒烟测试仅验证可运行（图形为视觉复刻，
不做数值对拍）。
