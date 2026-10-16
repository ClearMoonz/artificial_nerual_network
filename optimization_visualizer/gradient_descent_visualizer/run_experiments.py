"""Reproduce section 9.2 scenarios and export numeric evidence."""
from pathlib import Path
import json
from objectives import OBJECTIVES
from optimizers import Run, Settings, export_runs


def main():
    root=Path(__file__).resolve().parent.parent/'results'/'教材验证'
    cases=[]
    for a in [.01,.035,.039,.04,.041]:
        cases.append((f'A_alpha_{a}',[Run(OBJECTIVES['elongated'],'sd',[.5,.5],Settings('fixed',a))]))
    cases.append(('A_newton',[Run(OBJECTIVES['elongated'],'newton',[.5,.5])]))
    cases.append(('B_comparison',[Run(OBJECTIVES['rotated'],method,[.8,-.25]) for method in ['sd','newton','cg']]))
    for idx,start in enumerate([(1.5,0),(-1.5,0),(.75,.75),(1.15,.75)],6):
        cases.append((f'C_figure_9_{idx}',[Run(OBJECTIVES['nonconvex'],'newton',start)]))
    report=[]
    for name,runs in cases:
        for run in runs:
            run.finish()
        export_runs(root/name,runs)
        report.append({'case':name,'summaries':[r.summary() for r in runs]})
    (root/'验证汇总.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=['# 教材实验数值结果','',
          '| 实验 | 算法 | 步数 | 最终位置 | 梯度范数 | 状态 |',
          '|---|---|---:|---|---:|---|']
    for case in report:
        for s in case['summaries']:
            x=s['x']
            rows.append(f"| {case['case']} | {s['name']} | {s['steps']} | ({x[0]:.8g}, {x[1]:.8g}) | {s['gradient_norm']:.3e} | {s['status']} |")
    rows+=['','固定步长0.04位于稳定边界，0.041触发坐标保护；这两个结果不标记为收敛。',
           '默认梯度阈值1e-6、最大1000步、坐标上限1e6。所有CSV包含初始点；JSON包含逐步公式参数。',
           '来源为教材第9.2节；舍入后的教材数值不用于后续计算。']
    (root/'验证汇总.md').write_text('\n'.join(rows),encoding='utf-8')
    print('\n'.join(rows))


if __name__=='__main__': main()
