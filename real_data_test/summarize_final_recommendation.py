import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(__file__)


def robust_sigma(arr):
    arr = np.asarray(arr, float)
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad if mad > 0 else float(np.std(arr))


def summarize_one(path_csv, method_name, top_quantile=0.75):
    df = pd.read_csv(path_csv)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Q','Uc','a'])
    if len(df) == 0:
        return {'method': method_name, 'n': 0}
    Q = df['Q'].to_numpy(float)
    Q_med = float(np.median(Q)); Q_sig = robust_sigma(Q)
    q_thr = float(np.quantile(Q, top_quantile))
    high = df[df['Q'] >= q_thr].copy()
    n_high = len(high)
    if n_high == 0:
        return {'method': method_name, 'n': len(df)}
    Uc_med = float(np.median(high['Uc'])); Uc_sig = robust_sigma(high['Uc'])
    a_med  = float(np.median(high['a']));  a_sig  = robust_sigma(high['a'])
    return {
        'method': method_name,
        'n': len(df), 'n_high': n_high,
        'Q_median': Q_med, 'Q_sigma': Q_sig, 'Q_top_quantile': top_quantile, 'Q_threshold': q_thr,
        'Uc_median': Uc_med, 'Uc_robust_sigma': Uc_sig,
        'a_median': a_med, 'a_robust_sigma': a_sig,
    }


def write_final_report(nofse_stats, fse_stats, out_md):
    lines = []
    lines.append('# 最终综合结论与推荐参数\n')
    lines.append('## 方法与选择原则\n')
    lines.append('- 基于广义bootstrap结果，按质量上四分位（top 25%）筛选高质量子集进行统计；')
    lines.append('- 给出 `U_c` 与 `ν^{-1}` 的中位数作为推荐值，robust σ（≈1.4826·MAD）作为不确定度；')
    lines.append('- No-FSE Drop L=7 作为主结果，FSE All-L 作为对照；')
    lines.append('')
    lines.append('## 统计结果（高质量子集）\n')
    for st in [nofse_stats, fse_stats]:
        if not st or 'n' not in st or st['n'] == 0:
            continue
        lines.append(f"### {st['method']}\n")
        lines.append(f"- 样本数 n={st['n']}，高质量 n_high={st.get('n_high','NA')}（Q≥Q@{int(100*st.get('Q_top_quantile',0.75))}%={st.get('Q_threshold','NA'):.2f}）\n")
        lines.append(f"- 推荐 U_c = {st.get('Uc_median','NA'):.6f} ± {st.get('Uc_robust_sigma','NA'):.6f}\n")
        lines.append(f"- 推荐 ν^(-1) = {st.get('a_median','NA'):.6f} ± {st.get('a_robust_sigma','NA'):.6f}\n")
        lines.append('')
    lines.append('## 图表索引\n')
    lines.append('- 交点基线：`crossing_uc_extrapolation.png`')
    lines.append('- 初始敏感性：`sensitivity_nofse_dropL7.png`，`sensitivity_fse_allL.png`')
    lines.append('- 宽扫描与稳健扫描：`wide_a_scan_nofse_dropL7.png`，`robust_nofse_dropL7_scan.png`')
    lines.append('- FSE网格：`robust_fse_grid_heatmap.png`')
    lines.append('- 广义bootstrap：`generalized_bootstrap_nofse_dropL7.png`，`generalized_bootstrap_fse_allL.png`')
    lines.append('- 等效带对比：`equivalence_comparison.png`')
    lines.append('')
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    nofse_csv = os.path.join(BASE, 'generalized_bootstrap_nofse_dropL7.csv')
    fse_csv   = os.path.join(BASE, 'generalized_bootstrap_fse_allL.csv')
    nofse = summarize_one(nofse_csv, 'No-FSE (Drop L=7)', top_quantile=0.75)
    fse   = summarize_one(fse_csv, 'FSE (All L)', top_quantile=0.75)
    out_md = os.path.join(BASE, 'FINAL_SYNTHESIS_REPORT.md')
    write_final_report(nofse, fse, out_md)
    print('Wrote', os.path.basename(out_md))
    print('No-FSE stats:', nofse)
    print('FSE stats:', fse)

if __name__ == '__main__':
    main() 