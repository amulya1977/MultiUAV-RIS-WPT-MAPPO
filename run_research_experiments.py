"""Stage 8: controlled experiment runner, invoked from the repository root.
Run `python run_research_experiments.py --help`.
Measurements are generated from actual training and held-out evaluation; failures are retained.
"""
import argparse
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pls_mappo.environment import PLSMultiUAVEnv
from pls_mappo.config import SimulationConfig
from pls_mappo.evaluate_mappo import evaluate, load_actor, summarize


def get_cases(mode, grid):
    if mode == 'sensitivity':
        rates = [0.01, 0.03, 0.05]
        batches = [128, 256, 512]
        if grid == 'paired':
            cases = [dict(actor_lr=x, critic_lr=x, minibatch=b, uavs=3, iot=12, tx=0.005)
                     for b, x in itertools.product(batches, rates)]
        else:
            cases = [dict(actor_lr=a, critic_lr=c, minibatch=b, uavs=3, iot=12, tx=0.005)
                     for b, a, c in itertools.product(batches, rates, rates)]
        # A reference run is essential: 0.01-0.05 are high rates for PPO.
        cases += [dict(actor_lr=3e-5, critic_lr=1e-4, minibatch=b, uavs=3, iot=12, tx=0.005)
                  for b in batches]
        return cases
    if mode == 'scalability':
        sizes = [(2, 8), (3, 12), (4, 16), (4, 24)]
        return [dict(actor_lr=3e-5, critic_lr=1e-4, minibatch=256,
                     uavs=k, iot=n, tx=0.005) for k, n in sizes]
    if mode == 'traffic':
        # Transmit-power demand proxy; not MEC computational task size.
        return [dict(actor_lr=3e-5, critic_lr=1e-4, minibatch=256,
                     uavs=3, iot=12, tx=tx) for tx in (0.0005, 0.005, 0.01)]
    raise ValueError(mode)


def identity(c, seed):
    return f"K{c['uavs']}_N{c['iot']}_TX{c['tx']:g}_A{c['actor_lr']:g}_C{c['critic_lr']:g}_B{c['minibatch']}_S{seed}".replace('.', 'p')


def evaluate_checkpoint(path, c, args):
    ev_args = argparse.Namespace(episode_steps=args.episode_steps, num_uavs=c['uavs'],
        num_iot=c['iot'], iot_tx_power_w=c['tx'], collision_penalty=0.2,
        charging_deficit_penalty=0.0, target_harvest_j=0.0)
    device = torch.device('cpu')
    prototype = PLSMultiUAVEnv(config=SimulationConfig(num_uavs=c['uavs'],num_iot=c['iot']),
        episode_steps=args.episode_steps,requested_iot_tx_w=c['tx'],collision_penalty=0.2)
    actor = load_actor(path, prototype, device)
    rows = []
    for name, model in [('mappo', actor), ('random', None), ('stationary', None)]:
        for seed in range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes):
            row = evaluate(name, model, seed, ev_args, device)
            rows.append(row)
    return rows, summarize(rows, ('mappo','random','stationary'))


def save_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description='Stage 8 experimental measurements and tables')
    p.add_argument('--mode', choices=['sensitivity','scalability','traffic'], default='sensitivity')
    p.add_argument('--grid', choices=['paired','full'],default='paired',
                   help='paired: 3 LR pairs x 3 batches + 3 stable references; full: 27 LR pairs + 3 references')
    p.add_argument('--updates',type=int,default=2,
                   help='2 = pipeline smoke test, NOT publishable training evidence')
    p.add_argument('--rollout-steps',type=int,default=256,
                   help='At 3 UAVs gives 768 actor samples, enough for batch size 512')
    p.add_argument('--episode-steps',type=int,default=100)
    p.add_argument('--eval-episodes',type=int,default=2,
                   help='Use >=10 for research analysis; 2 is pipeline smoke test')
    p.add_argument('--eval-seed-start',type=int,default=3000)
    p.add_argument('--train-seeds',type=int,nargs='+',default=[42],
                   help='Use 42 43 44 for independent-run reliability')
    p.add_argument('--output-dir',default='pls_mappo/research_results')
    p.add_argument('--max-cases',type=int,default=0,help='Optional limit for quick verification')
    args=p.parse_args()
    if min(args.updates,args.rollout_steps,args.episode_steps,args.eval_episodes,len(args.train_seeds)) < 1:
        p.error('Count arguments must be positive')
    cases=get_cases(args.mode,args.grid)
    if args.max_cases:
        cases=cases[:args.max_cases]
    root=Path(args.output_dir)/args.mode
    root.mkdir(parents=True, exist_ok=True)
    all_summary=[]; all_episodes=[]
    for i,c in enumerate(cases, 1):
        for train_seed in args.train_seeds:
            tag=identity(c,train_seed)
            target=root/tag
            target.mkdir(parents=True,exist_ok=True)
            checkpoint=target/f'mappo_update_{args.updates:05d}.pt'
            cmd=[sys.executable,'-m','pls_mappo.train_mappo',
                 '--updates',str(args.updates),'--rollout-steps',str(args.rollout_steps),
                 '--episode-steps',str(args.episode_steps),'--minibatch-size',str(c['minibatch']),
                 '--actor-lr',str(c['actor_lr']),'--critic-lr',str(c['critic_lr']),
                 '--iot-tx-power-w',str(c['tx']), '--num-uavs',str(c['uavs']),
                 '--num-iot',str(c['iot']),'--collision-penalty','0.2',
                 '--charging-deficit-penalty','0', '--target-harvest-j','0',
                 '--save-every',str(args.updates), '--seed',str(train_seed),
                 '--output-dir',str(target)]
            print(f'[{i}/{len(cases)}] {tag}',flush=True)
            start=time.time()
            meta=dict(mode=args.mode,case_id=tag,actor_lr=c['actor_lr'],critic_lr=c['critic_lr'],
                      minibatch_size=c['minibatch'],num_uavs=c['uavs'],num_iot=c['iot'],
                      requested_tx_w=c['tx'],train_seed=train_seed,updates=args.updates,
                      rollout_steps=args.rollout_steps,eval_episodes=args.eval_episodes)
            try:
                if not checkpoint.exists():
                    proc=subprocess.run(cmd, text=True, capture_output=True, check=False)
                    (target/'train_stdout.txt').write_text(proc.stdout,encoding='utf-8')
                    (target/'train_stderr.txt').write_text(proc.stderr,encoding='utf-8')
                    if proc.returncode != 0:
                        raise RuntimeError(f'training exit code {proc.returncode}: {proc.stderr[-450:]}')
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                episodes, summary = evaluate_checkpoint(checkpoint,c,args)
                for row in episodes:
                    all_episodes.append(dict(meta, **row))
                for name, metrics in summary.items():
                    entry=dict(meta,policy=name,status='ok',seconds=round(time.time()-start,2))
                    for key,metric in metrics.items():
                        entry[key+'_mean']=metric['mean']; entry[key+'_std']=metric['std']
                    all_summary.append(entry)
                print('  OK: ' + ' | '.join(f"{x}: {summary[x]['secrecy_mbps']['mean']:.3f} Mbps" for x in summary),flush=True)
            except Exception as exc:
                fail=dict(meta,policy='mappo',status='failed',error=str(exc),seconds=round(time.time()-start,2))
                all_summary.append(fail)
                (target/'failure.txt').write_text(str(exc),encoding='utf-8')
                print(f'  FAILED: {exc}',flush=True)
            save_csv(root/'summary.csv',all_summary)
            save_csv(root/'episodes.csv',all_episodes)
    (root/'experiment_config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    print(f'\nSaved real results in {root}/summary.csv and episodes.csv')
    print('NOTE: 2-update runs are pipeline tests, not convergence/research evidence.')

if __name__=='__main__':
    main()
