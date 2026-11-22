"""Generate report Figures 4.3, 4.4, 4.5 using actual Othello project records.

Run from the folder containing main.py:
  python -m pip install matplotlib
  python generate_chapter4_figures.py

Requires:
  results/mcts/mcts_tournament_results.csv
  EXPERIMENT_RESULTS_CNN.md

Outputs:
  report_figures/hinh_4_3_mcts_thoi_gian.png
  report_figures/hinh_4_4_cnn_heuristic.png
  report_figures/hinh_4_5_cnn_validation_loss.png
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MCTS_CSV = ROOT / 'results' / 'mcts' / 'mcts_tournament_results.csv'
CNN_MD = ROOT / 'EXPERIMENT_RESULTS_CNN.md'
OUT = ROOT / 'report_figures'
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'figure.dpi': 120,
    'savefig.dpi': 300,
})


def save_chart(fig, filename):
    fig.tight_layout(pad=2.1)
    path = OUT / filename
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', path)


def parse_table_section(md, header):
    m = re.search(r'^##\s+' + re.escape(header) + r'\s*$', md, re.M)
    if not m:
        raise ValueError(f'Không tìm thấy mục "{header}" trong {CNN_MD}')
    part = md[m.end():]
    next_section = re.search(r'^##\s+', part, re.M)
    if next_section:
        part = part[:next_section.start()]
    rows = []
    for ln in part.splitlines():
        if not ln.startswith('|'):
            continue
        cells = [x.strip() for x in ln.strip().strip('|').split('|')]
        if any(not x.replace(':', '').replace('-', '').replace(' ', '') for x in cells):
            continue
        rows.append(cells)
    return rows


def score(wdl):
    m = re.fullmatch(r'(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', wdl)
    if not m:
        return None
    w,d,l = map(int,m.groups())
    n = w+d+l
    return (100*(w+0.5*d)/n, n, (w,d,l))


def get_mcts():
    if not MCTS_CSV.is_file():
        raise FileNotFoundError(f'Thiếu file: {MCTS_CSV}')
    with MCTS_CSV.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    # Only combine rows from a single complete run, not different experiments.
    group = {}
    for r in rows:
        rid = r['run_id']
        if rid not in group:
            group[rid] = {}
        group[rid][int(r['simulations'])] = float(r['average_decision_time_seconds'])
    candidates = [rid for rid, d in group.items() if all(k in d for k in (100,500,1000))]
    if not candidates:
        raise ValueError('CSV chưa có một run_id hoàn chỉnh cho 100/500/1000 simulations')
    latest = sorted(candidates)[-1]
    return latest, [group[latest][k] for k in (100,500,1000)]


def draw_43():
    run_id, ys = get_mcts()
    xs = [100, 500, 1000]
    fig, ax = plt.subplots(figsize=(8.6,5.2))
    bars = ax.bar([str(x) for x in xs], ys, width=.55)
    ax.set_ylim(0, max(ys)*1.23)
    ax.set_xlabel('Ngân sách mô phỏng (simulations/nước)')
    ax.set_ylabel('Thời gian suy luận trung bình (giây/nước)')
    ax.set_title('Thời gian suy luận MCTS theo ngân sách mô phỏng')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ys):
        ax.text(bar.get_x()+bar.get_width()/2, value+max(ys)*.025,
                f'{value:.4f}'.replace('.',','), ha='center', va='bottom', fontsize=11)
    fig.text(.52,.005, f'Nguồn: mcts_tournament_results.csv | run_id: {run_id}',
             ha='center', fontsize=8, color='#555555')
    save_chart(fig, 'hinh_4_3_mcts_thoi_gian.png')


def draw_44(md):
    rows = parse_table_section(md, 'So sánh với Heuristic')
    model_names, rates, sizes = [],[],[]
    for r in rows:
        if len(r) < 6: continue
        label = r[0].replace('*','').strip()
        s42, s123 = score(r[1].replace('*', '')), score(r[3].replace('*', ''))
        if not s42: continue
        w,d,l = s42[2]
        n = s42[1]
        if s123:
            w2,d2,l2 = s123[2]
            w,d,l = w+w2,d+d2,l+l2
            n += s123[1]
        model_names.append(label)
        rates.append(100*(w+.5*d)/n)
        sizes.append(n)
    if not model_names:
        raise ValueError('Không tìm thấy dữ liệu Heuristic trong bảng')
    fig, ax = plt.subplots(figsize=(10.3,5.6))
    tick_labels=[f'{name}\n(n={n})' for name,n in zip(model_names,sizes)]
    bars = ax.bar(tick_labels, rates, width=.66)
    ax.set_ylim(0,max(60,max(rates)+10))
    ax.set_ylabel('Score Rate (%)')
    ax.set_title('So sánh CNN trước Heuristic Agent')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for bar, value in zip(bars,rates):
        ax.text(bar.get_x()+bar.get_width()/2, value+1.05,
                f'{value:.2f}%'.replace('.',','), ha='center', fontsize=10)
    fig.text(.52,.002, 'n = số trận đã ghi nhận',
             ha='center', fontsize=9, color='#555555')
    save_chart(fig,'hinh_4_4_cnn_heuristic.png')


def draw_45(md):
    rows = parse_table_section(md, 'Đối chiếu checkpoint và Validation')
    names, loss, epochs = [],[],[]
    for r in rows:
        if len(r)<4: continue
        label = r[0].replace('*','').strip()
        try: v=float(r[3].replace(',','.')); epoch=int(r[2])
        except ValueError: continue
        names.append(label)
        loss.append(v)
        epochs.append(epoch)
    if not names:
        raise ValueError('Không tìm thấy bảng checkpoint/validation loss')
    fig, ax = plt.subplots(figsize=(9.0,5.4))
    bars=ax.bar(names,loss,width=.62)
    ax.set_ylim(0,max(2.2,max(loss)*1.13))
    ax.set_ylabel('Validation Loss tại checkpoint được chọn')
    ax.set_title('Validation Loss tốt nhất theo từng phiên bản CNN')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    for bar, value, epoch in zip(bars,loss,epochs):
        ax.text(bar.get_x()+bar.get_width()/2, value+.04,
                f'{value:.4f}\n(epoch {epoch})'.replace('.',','),
                ha='center', va='bottom', fontsize=9)
    save_chart(fig,'hinh_4_5_cnn_validation_loss.png')


if __name__=='__main__':
    if not CNN_MD.is_file():
        raise FileNotFoundError(f'Thiếu file: {CNN_MD}')
    md = CNN_MD.read_text(encoding='utf-8-sig')
    draw_43()
    draw_44(md)
    draw_45(md)
