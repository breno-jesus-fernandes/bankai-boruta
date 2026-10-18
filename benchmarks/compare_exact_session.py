"""Sequential fresh-process a6/a15 versus experimental exact session.

RSS is sampled externally only during fit. Linux VmHWM reset is attempted;
its success and resulting scope are recorded explicitly. Fixed rounds and complete fits are distinct experiment IDs.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

PROTOCOL = 'exact-session-a7-v1'


def worker(args):
    import numpy as np
    from bankai_boruta import BankaiBoruta, _boruta_core
    from bankai_random_forest import BankaiRandomForestClassifier
    if args.benchmark_root:
        sys.path.insert(0, str(Path(args.benchmark_root).resolve() / 'src'))
        from bankai_benchmark.datasets import generate_dataset
        bundle = generate_dataset('industrial', args.seed, 'full')
        X, y = bundle.X_train, bundle.y_train
        dataset = bundle.metadata
        del bundle
    else:
        rng = np.random.RandomState(args.seed)
        X = rng.normal(size=(args.rows,args.columns)).astype(args.dtype)
        y = (X[:,0] + .4*X[:,1] > 0).astype(int)
        dataset = {'generator':'controlled-normal-v1','shape':list(X.shape),'dtype':str(X.dtype)}
    estimator = BankaiRandomForestClassifier(n_estimators=args.trees, max_depth=6,
        extra_trees=True, extra_trees_backend='exact', bootstrap=True,
        max_features='sqrt', n_jobs=args.threads)
    selector = BankaiBoruta(estimator,n_estimators=None, random_state=args.seed,
        max_iter=args.fixed_rounds+1 if args.fixed_rounds else args.max_iter,
        early_stopping=True,n_iter_no_change=100,store_history=True)
    target=Path(args.worker_output)
    target.with_suffix('.ready').touch()
    while not target.with_suffix('.go').exists(): time.sleep(.001)
    hwm_reset=False
    if sys.platform.startswith('linux'):
        try:
            Path('/proc/self/clear_refs').write_text('5')
            hwm_reset=True
        except OSError: pass
    start = time.perf_counter()
    selector.fit(X,y)
    elapsed = time.perf_counter()-start
    hwm = None
    if sys.platform.startswith('linux'):
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmHWM:'): hwm=int(line.split()[1])*1024
    target.with_suffix('.done').touch()
    result = dict(protocol=PROTOCOL, elapsed_seconds=elapsed, diagnostics=selector.diagnostics_,
        n_iter=selector.n_iter_,stop_reason=selector.stop_reason_,dataset=dataset,
        data_sha256=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest(),
        vmhwm_bytes=hwm,vmhwm_reset=hwm_reset,vmhwm_scope='fit' if hwm_reset else 'process_lifetime',
        source_hashes={'native_binary':hashlib.sha256(Path(_boruta_core.__file__).read_bytes()).hexdigest(),
            'selector':hashlib.sha256(Path(sys.modules[BankaiBoruta.__module__].__file__).read_bytes()).hexdigest()},
        python_version=sys.version,
        packages={k:importlib.metadata.version(k) for k in ['bankai-boruta','bankai-random-forest','numpy','scipy','scikit-learn']},
        hits=selector.hits_.tolist(),support=selector.support_.tolist(),weak=selector.support_weak_.tolist(),ranking=selector.ranking_.tolist(),
        importances=np.where(np.isnan(selector.importance_history_),None,selector.importance_history_).tolist(),
        thresholds=selector.shadow_threshold_history_.tolist())
    Path(args.worker_output).write_text(json.dumps(result,allow_nan=False))


def run(python, variant, args):
    import psutil
    with tempfile.TemporaryDirectory() as temp:
        target=Path(temp)/'result.json'
        cmd=[python,str(Path(__file__).resolve()),'--worker-output',str(target),
            '--rows',str(args.rows),'--columns',str(args.columns),'--trees',str(args.trees),
            '--threads',str(args.threads),'--seed',str(args.seed),'--dtype',args.dtype,
            '--fixed-rounds',str(args.fixed_rounds),'--max-iter',str(args.max_iter)]
        if args.benchmark_root: cmd += ['--benchmark-root',args.benchmark_root]
        env=os.environ.copy()
        env.pop('PYTHONPATH',None)
        env['BANKAI_BORUTA_EXACT_SESSION']='1' if variant == 'session' else '0'
        for key in ['OMP_NUM_THREADS','RAYON_NUM_THREADS']: env[key]=str(args.threads)
        for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: env[key]='1'
        # Avoid inherited editable source resolution through the working directory.
        with (Path(temp)/'stderr.log').open('w+') as errors:
            proc=subprocess.Popen(cmd,cwd=temp,env=env,stdout=errors,stderr=errors)
            monitored=psutil.Process(proc.pid); peak=0; baseline=None; deadline=time.monotonic()+args.timeout_seconds
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    proc.terminate()
                    try: proc.wait(timeout=5)
                    except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                    raise TimeoutError(f'{variant} exceeded {args.timeout_seconds}s process deadline')
                if target.with_suffix('.ready').exists() and baseline is None:
                    baseline=monitored.memory_info().rss
                    peak=baseline
                    target.with_suffix('.go').touch()
                if baseline is not None and not target.with_suffix('.done').exists():
                    try: peak=max(peak,monitored.memory_info().rss)
                    except psutil.NoSuchProcess: pass
                time.sleep(.005)
            if proc.returncode:
                errors.seek(0); raise RuntimeError(errors.read())
        result=json.loads(target.read_text()); result.update(variant=variant,external_peak_rss_bytes=peak,rss_scope='fit',baseline_rss_bytes=baseline,peak_rss_delta_bytes=peak-baseline)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-python',default=sys.executable)
    parser.add_argument('--candidate-python',default=sys.executable)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker-output')
    parser.add_argument('--benchmark-root',help='Use the frozen benchmark Industrial full generator; overrides controlled dimensions')
    parser.add_argument('--rows',type=int,default=20000)
    parser.add_argument('--columns',type=int,default=100)
    parser.add_argument('--trees',type=int,default=25)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--dtype',choices=['float32','float64'],default='float32')
    parser.add_argument('--fixed-rounds',type=int,default=5,help='0 selects complete fits')
    parser.add_argument('--max-iter',type=int,default=120)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout-seconds',type=float,default=1800,help='Hard deadline for each subprocess, including data preparation')
    args=parser.parse_args()
    if args.timeout_seconds <= 0 or args.repeats < 1: parser.error('positive timeout and repeats are required')
    if args.worker_output: worker(args); return
    if not args.output: parser.error('--output is required')
    if args.output.exists(): parser.error('output already exists; use a distinct experiment path')
    import numpy as np
    results=[]
    for repeat in range(args.repeats):
        pair=[]
        # Alternate which process runs first while keeping execution sequential.
        variants=['reference','session'] if repeat%2 == 0 else ['session','reference']
        for variant in variants:
            result=run(args.reference_python if variant=='reference' else args.candidate_python,variant,args)
            result['repeat']=repeat;pair.append(result);results.append(result)
            print(f"{variant} repeat={repeat}: {result['elapsed_seconds']:.3f}s RSS={result['external_peak_rss_bytes']/2**20:.1f}MiB",flush=True)
        a,b=pair
        reference=next(r for r in pair if r['variant']=='reference')
        if (reference['packages']['bankai-boruta'],reference['packages']['bankai-random-forest']) != ('0.1.0a6','0.1.0a15'):
            raise AssertionError('reference must be frozen a6/a15')
        for name in ['data_sha256','hits','support','weak','ranking','n_iter','stop_reason']:
            if a[name]!=b[name]: raise AssertionError(f'parity failed: {name}')
        for name in ['importances','thresholds']:
            np.testing.assert_allclose(np.asarray(a[name],float),np.asarray(b[name],float),rtol=1e-12,atol=1e-12,equal_nan=True)
        if args.fixed_rounds and a['n_iter']!=args.fixed_rounds:
            raise AssertionError('decided before fixed round budget; choose another controlled fixture')
    summary={variant:{key:statistics.median(r[key] for r in results if r['variant']==variant)
        for key in ['elapsed_seconds','external_peak_rss_bytes']} for variant in ['reference','session']}
    for variant in summary:
        for key in ['elapsed_seconds','external_peak_rss_bytes']:
            q=np.quantile([r[key] for r in results if r['variant']==variant],[.25,.75])
            summary[variant][key+'_q25']=float(q[0]); summary[variant][key+'_q75']=float(q[1])
    artifact=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol=PROTOCOL,experiment_kind='fixed_rounds' if args.fixed_rounds else 'complete_fit',
        host=dict(platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count()),
        configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        parity='passed',summary=summary,results=results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
