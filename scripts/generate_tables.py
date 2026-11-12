"""Rebuild manuscript tables from the supplied criterion-level result exports."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / 'build' / 'tables'
DATA = ROOT / 'data' / 'results'
CONFIGS = [('gpt_full_context', 'GPT full context'), ('gpt_rag', 'GPT RAG'),
           ('gemini_full_context', 'Gemini full context'), ('gemini_rag', 'Gemini RAG')]

def read(name):
    with (DATA / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def number(value):
    value = float(value)
    return f'{0.0 if abs(value) < 0.0005 else value:.3f}'

def write(name, text):
    (TABLES / name).write_text(text.strip() + '\n', encoding='utf-8')

def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = read('criterion_results.csv')
    pairs = read('full_context_rag_comparisons.csv')
    indexed = {(r['config_id'], r['question_id']): r for r in rows}
    lines = [r'\begin{table}[tbp]', r'\centering',
             r'\caption{Selected criterion-level model performance against the expert mean.}',
             r'\label{tab:selected-criteria}', r'\small',
             r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Configuration & Pearson $r$ & Spearman $\rho$ & MAE & RMSE & Bias \\', r'\midrule']
    selected = [('Q005', 'Environmental problem as a priority'),
                ('Q017', 'Environmental benefit in the business model'),
                ('Q026', 'Minimizing negative economic effects'),
                ('Q033', 'Team skills for sustainability goals')]
    for i, (qid, title) in enumerate(selected):
        if i:
            lines.append(r'\addlinespace[4pt]')
        lines.append(r'\multicolumn{6}{l}{\textit{' + qid + ': ' + title + r'}} \\')
        for cid, label in CONFIGS:
            row = indexed[cid, qid]
            lines.append(label + ' & ' + ' & '.join(number(row[m]) for m in
                          ['pearson', 'spearman', 'mae', 'rmse', 'bias']) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip', r'\end{table}']
    write('selected_criteria.tex', '\n'.join(lines))
    lines = [r'\begin{table}[tbp]', r'\centering',
             r'\caption{Criterion level model disagreement relative to expert disagreement.}',
             r'\label{tab:human-relative}', r'\small',
             r'\begin{tabular}{lcc}', r'\toprule',
             r'Configuration & \shortstack{Lower disagreement\\observed dataset} & \shortstack{Lower disagreement\\supported by 95\% CI} \\', r'\midrule']
    for cid, label in CONFIGS:
        sub = [r for r in rows if r['config_id'] == cid]
        point = sum(float(r['hrmae']) > 1 for r in sub)
        interval = sum(float(r['hrmae_ci_low']) > 1 for r in sub)
        lines.append(f'{label} & {point}/37 & {interval}/37' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip', r'\end{table}']
    write('human_relative_summary.tex', '\n'.join(lines))
    lines = [r'\begin{table}[tbp]', r'\centering',
             r'\caption{Paired full context and RAG error comparisons across 37 criteria.}',
             r'\label{tab:method-comparison}', r'\small', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{lccccc}', r'\toprule',
             r'& \multicolumn{3}{c}{Lower MAE in observed data} & \multicolumn{2}{c}{95\% CI supports} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}',
             r'Family & Full context & RAG & Tie & Full context & RAG \\', r'\midrule']
    for family, label in [('gpt', 'GPT'), ('gemini', 'Gemini')]:
        sub = [r for r in pairs if r['comparison_id'] == family]
        values = [sum(float(r['delta_mae']) > 0 for r in sub),
                  sum(float(r['delta_mae']) < 0 for r in sub),
                  sum(float(r['delta_mae']) == 0 for r in sub),
                  sum(float(r['delta_mae_ci_low']) > 0 for r in sub),
                  sum(float(r['delta_mae_ci_high']) < 0 for r in sub)]
        lines.append(label + ' & ' + ' & '.join(map(str, values)) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\par\smallskip', r'\end{table}']
    write('method_summary.tex', '\n'.join(lines))
    print('Wrote three manuscript tables.')

if __name__ == '__main__':
    main()
