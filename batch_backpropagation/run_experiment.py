"""Reproducible CLI experiments and exported CSV/JSON results."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from nn_sandbox.backend.algorithms.batch_bp import BatchBP, Config, xor_data, parse_data


def main():
    p = argparse.ArgumentParser(description='习题4.5：批训练 BP')
    p.add_argument('--activation', choices=['unipolar', 'bipolar'], default='unipolar')
    p.add_argument('--inputs', type=int, default=2)
    p.add_argument('--hidden', type=int, default=4)
    p.add_argument('--outputs', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=0.8)
    p.add_argument('--threshold', type=float, default=0.05)
    p.add_argument('--max-epochs', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data', type=Path, help='无表头 CSV/文本数据，输入在前，目标在后')
    p.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    a = p.parse_args()
    cfg = Config(a.inputs, a.hidden, a.outputs, a.learning_rate, a.activation,
                 a.threshold, a.max_epochs, a.seed)
    x, y = parse_data(a.data.read_text(encoding='utf-8-sig'), a.inputs, a.outputs) if a.data else xor_data(a.activation)
    model = BatchBP(cfg)
    model.fit(x, y)
    a.output.mkdir(parents=True, exist_ok=True)
    prefix = a.output / a.activation
    with prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'error', 'rmse'])
        writer.writeheader(); writer.writerows(model.history)
    result = {'config': vars(cfg), 'status': model.status, 'final': model.history[-1],
              'inputs': x.tolist(), 'targets': y.tolist(), 'predictions': model.predict(x).tolist()}
    prefix.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez(prefix.with_suffix('.npz'), V=model.V, W=model.W)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if model.status == '达到误差阈值' else 2


if __name__ == '__main__':
    raise SystemExit(main())
