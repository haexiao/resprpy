# resprpy（中文说明）

**resprpy** 是 R 包 **respR 2.3.4**（Harianto & Carey, *Methods in Ecology
and Evolution* 2019, doi:10.1111/2041-210X.13162）的完整 Python 移植，
用于水生态呼吸代谢（耗氧率/代谢率）数据的处理与分析。

目标是**与 R 数值完全一致**：全部导出函数均与 R 4.5.3 实测输出逐值对拍
（pytest 对拍测试 57 项全绿）。

> English README: [README.md](README.md) ·
> 使用手册（中文）: [docs/respR_resprpy_guide.zh.html](docs/respR_resprpy_guide.zh.html) ·
> Usage guide (EN): [docs/respR_resprpy_guide.en.html](docs/respR_resprpy_guide.en.html)

## 安装

```bash
pip install -e .            # 或 pip install -e ".[plot,test]"
```

依赖：numpy、scipy（可选 matplotlib、pytest、openpyxl）。

## 功能覆盖（respR 全部 27 个导出函数）

| respR (R) | resprpy (Python) | 状态 |
|---|---|---|
| `calc_rate` | `calc_rate` | ✅ ≤1e-13 |
| `calc_rate.int` | `calc_rate_int` | ✅ ≤1e-6 |
| `calc_rate.bg` | `calc_rate_bg` | ✅ ≤1e-6 |
| `calc_rate.ft` | `calc_rate_ft` | ✅ ≤1e-5 |
| `auto_rate` | `auto_rate` | ✅ bw.SJ/density 逐位一致 |
| `auto_rate.int` | `auto_rate_int` | ✅ ≤1e-5 |
| `adjust_rate` | `adjust_rate` | ✅ 6 种方法 |
| `adjust_rate.ft` | `adjust_rate_ft` | ✅ ≤1e-6 |
| `select_rate` | `select_rate` | ✅ 34 种方法全覆盖 |
| `select_rate.ft` | `select_rate_ft` | ✅ ≤1e-5 |
| `convert_DO` | `convert_DO` | ✅ ≤1e-12 |
| `convert_MR` | `convert_MR` | ✅ ≤1e-12 |
| `convert_rate` | `convert_rate` | ✅ ≤1e-12 |
| `convert_rate.ft` | `convert_rate_ft` | ✅ ≤1e-8 |
| `convert_val` | `convert_val` | ✅ ≤1e-12 |
| `format_time` | `format_time` | ✅ 逐值一致 |
| `inspect` | `inspect` | ✅ checks/locs 一致 |
| `inspect.ft` | `inspect_ft` | ✅ |
| `oxy_crit` | `oxy_crit` | ✅ bsr + segmented |
| `import_file` | `import_file` | ✅ NeoFox 逐值（14 解析器） |
| `sim_data` | `sim_data` | ✅ 结构一致 |
| `subsample` | `subsample` | ✅ |
| `subset_data` | `subset_data` | ✅ |
| `select` | `select` | ✅ |
| `%>%` | （无，Python 链式调用） | ➖ 不适用 |
| `test_lin` | `test_lin` | 🟡 结构一致 |
| `unit_args` | `unit_args` | ✅ |
| S3 `plot()` | `plot_inspect` `plot_calc_rate` `plot_auto_rate` `plot_oxy_crit` | ✅ matplotlib |

## 亮点

- **Segmented（Muggeo 2003）断点分析**：移植自 `segmented` 2.2-1 包，
  **包含 R 的 MT19937 随机数引擎逐位复刻**（set.seed 序列一致、
  bootstrap-restart 搜索），crit 与 R 逐位相同（如 7.719111…）。
- **`select_rate`**：34 种方法全覆盖对拍，含错误行为（R 拒绝的方法
  Python 报同样的错）。
- **`convert_*`**：物理锚点全中（sw_dens=1024.64077347883、
  molvol=24.03038425、mg/L→%Air=108.4419）。
- **无 C/C++ 运行时**：respR 依赖的滚动回归、核密度、marelac 物理常数、
  segmented 算法全部内联为纯 numpy/scipy——方便 PyInstaller 打包。

## 测试

```bash
python -m pytest tests/ -v        # 57 项，全绿
```

`tests/reference/` 下的参考 CSV 由 R 4.5.3 实测生成
（`reference/gen_refs_*.R`）；pytest 逐元素对比（容差分级：回归系数
1e-8、rsq 1e-12、time 1e-9、density 1e-3）。

## 包结构对照（vs respR）

详见使用手册中的"结构对照"章节。

## 许可

GPL-3（与 respR 相同）。本项目是 respR 及其依赖算法（roll/marelac/
segmented）的移植实现，发布须遵守 GPL-3。
