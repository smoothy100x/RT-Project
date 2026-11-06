from __future__ import annotations;import concurrent.futures as cf, hashlib, html;import json, logging, math, os, re, sys, time, traceback, tempfile, threading;from pathlib import Path;from types import SimpleNamespace;os.environ.setdefault('OMP_NUM_THREADS', '2');os.environ.setdefault('OPENBLAS_NUM_THREADS', '2');os.environ.setdefault('MKL_NUM_THREADS', '2');os.environ.setdefault('MPLCONFIGDIR', '/tmp/fi2010_matplotlib');import numpy as np;import pandas as pd;import scipy.linalg as la;from scipy import stats;import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt;from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix;import torch
try:import psutil
except ImportError:psutil = None
SETTINGS = {'input': '/kaggle/input/datasets/alpha000x/fi-2010/BenchmarkDatasets', 'output': '/kaggle/working/fi2010_results', 'hours': 8.0, 'purge': 100, 'seed': 2026, 'max_deletions': -1, 'primary_only': False};HORIZONS = [10, 20, 30, 50, 100];CLASSES = ['up', 'stationary', 'down'];COURSE = {'W01-W04': 'Linear-in-parameters models, least squares, projection geometry, rank and normal theory', 'W05': 'Nested-model reference F tests, R squared, confidence ellipsoids', 'W06': 'Bonferroni and Scheffe simultaneous reference intervals and linear contrasts', 'W07': 'Reference regression-mean intervals; prediction intervals inappropriate as class probabilities', 'W08-W09': 'Omitted variables, dependence, unequal variance and model-assumption audits', 'W10': 'Condition indices, VIF, Mahalanobis/leverage identity, outliers and influence', 'W11': 'PRESS, externally studentized residuals, DFBETAS, DFFITS, COVRATIO and Cook distance', 'W12': 'Partial residual plots, polynomial terms, feasible WLS, AR(1) feasible GLS'};WARNINGS = ['All t/F p-values and confidence intervals are Gaussian IID reference quantities only. Discrete labels violate normality; overlapping horizons and adjacent books violate independence.', 'Direction score is an ordinal coding, not a continuous return. Indicator regressions are linear probability models: scores may lie outside [0,1] and are not calibrated probabilities.', 'No stock IDs or reliable within-day stock boundaries are supplied in the benchmark matrices. No synthetic lagged features, monetary backtest, spread reconstruction, or trade-level interpretation is attempted.', 'Upstream benchmark normalization fitting scope is not verified. Additional scaling, rank selection, basis selection and variance fitting use training rows only.', 'All fold copies and normalization copies describe overlapping observations. They are never pooled as independent samples.', 'Block deletion is a sensitivity study, not a bootstrap confidence interval. Nominal significance is not proof of an edge.'];LOG = logging.getLogger('fi2010')
def dump(path, obj):
 path = Path(path);path.parent.mkdir(parents=True, exist_ok=True);tmp = path.with_suffix(path.suffix + '.tmp')
 def conv(x):
  if isinstance(x, np.ndarray):return x.tolist()
  if isinstance(x, np.generic):return x.item()
  if isinstance(x, Path):return str(x)
  raise TypeError(type(x).__name__)
 tmp.write_text(json.dumps(obj, indent=2, default=conv, allow_nan=True));tmp.replace(path)
def table(path, rows):Path(path).parent.mkdir(parents=True, exist_ok=True);pd.DataFrame(rows).to_csv(path, index=False)
def fig(path, title, draw):f, ax = plt.subplots(figsize=(10, 5));draw(ax);ax.set_title(title);f.tight_layout();f.savefig(path, dpi=140);plt.close(f)
def heatmap(ax, matrix, xlabels=None, ylabels=None, **kw):
 artist = ax.imshow(matrix, aspect='auto', **kw);ax.figure.colorbar(artist, ax=ax, pad=0.02)
 if xlabels is not None:ax.set_xticks(np.arange(len(xlabels)), xlabels, rotation=90 if len(xlabels) > 7 else 0, fontsize=7 if len(xlabels) > 7 else 10)
 if ylabels is not None:ax.set_yticks(np.arange(len(ylabels)), ylabels)
 return artist
def feature_names():
 names = []
 for level in range(1, 11):names += [f'ask_price_L{level}', f'ask_volume_L{level}', f'bid_price_L{level}', f'bid_volume_L{level}']
 return names + [f'engineered_{i + 1:03d}' for i in range(40, 144)]
def view_key(p):s = str(p).lower();auction = 'NoAuction' if 'noauction' in s else 'Auction';norm = 'Zscore' if 'zscore' in s or 'z-score' in s else 'MinMax' if 'minmax' in s else 'DecPre' if 'decpre' in s else None;return (auction, norm) if norm else None
def discover(root, out):
 files = sorted((p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in {'.txt', '.csv', '.npy'}));inventory = [];groups = {}
 for p in files:
  key = view_key(p);m = re.search('CF[_-]?(\\d+)', p.stem, re.I);role = 'train' if 'train' in p.stem.lower() else 'test' if 'test' in p.stem.lower() else None;inventory.append(dict(path=str(p), bytes=p.stat().st_size, view=str(key), fold=int(m[1]) if m else None, role=role))
  if key and m and role:
   slot = groups.setdefault(key, {}).setdefault(int(m[1]), {})
   if role in slot:raise ValueError(f'Ambiguous duplicate files for {key}, CF{m[1]}, {role}; narrow SETTINGS input')
   slot[role] = p
 table(out / 'file_inventory.csv', inventory);good = {k: v for k, v in groups.items() if 9 in v and {'train', 'test'} <= v[9].keys()}
 if set(groups) - set(good):raise ValueError(f'Incomplete CF9 views: {set(groups) - set(good)}; mount all train/test files')
 if not good:raise ValueError('No complete FI-2010 CF9 train/test pair. Expected Auction/NoAuction + Zscore/MinMax/DecPre and *Train*CF_9.txt, *Test*CF_9.txt. See file_inventory.csv; mount the extracted dataset, not an archive.')
 return good
def first_width(p):
 if p.suffix == '.npy':a = np.load(p, mmap_mode='r');return a.shape[1] if a.shape[0] == 149 else a.shape[0]
 with p.open() as f:
  for line in f:
   if line.strip():return len(line.replace(',', ' ').split())
 raise ValueError(f'Empty file {p}')
def read_matrix(p):
 if p.suffix == '.npy':a = np.load(p)
 else:
  width = first_width(p)
  if width == 149:a = np.loadtxt(p, delimiter=',' if p.suffix == '.csv' else None, dtype=np.float64)
  else:
   a = np.empty((149, width), np.float64);row = 0
   with p.open() as f:
    for line in f:
     if not line.strip():continue
     if row >= 149:raise ValueError(f'{p}: more than 149 feature/label rows')
     vals = np.fromstring(line.replace(',', ' '), sep=' ')
     if len(vals) != width:raise ValueError(f'{p}: ragged/non-numeric row {row}: {len(vals)} != {width}')
     a[row] = vals;row += 1
   if row != 149:raise ValueError(f'{p}: expected 149 rows, got {row}')
 if a.ndim != 2:raise ValueError(f'{p}: not a matrix')
 if a.shape[0] == 149:a = a.T
 if a.shape[1] != 149:raise ValueError(f'{p}: expected 144 features + 5 labels; shape={a.shape}')
 if not np.isfinite(a).all():raise ValueError(f'{p}: missing or infinite values (not silently dropped)')
 labels = a[:, 144:]
 if not np.isin(labels, [1, 2, 3]).all():raise ValueError(f'{p}: labels must be original FI codes 1,2,3, in last five columns')
 return (np.ascontiguousarray(a[:, :144]), labels.astype(np.int8) - 1)
def load_view(folds, out):
 x, y = read_matrix(folds[9]['train']);xt, yt = read_matrix(folds[9]['test']);n = len(x);counts = {9: n}
 for k in range(1, 9):
  if k in folds and 'train' in folds[k]:counts[k] = first_width(folds[k]['train'])
 exact = set(counts) == set(range(1, 10)) and all((counts[k] > counts[k - 1] for k in range(2, 10)))
 if exact:bounds = [0] + [counts[k] for k in range(1, 10)] + [n + len(xt)]
 else:bounds = [0, n, n + len(xt)]
 dump(out / 'data_audit.json', dict(train_rows=n, test_rows=len(xt), total_rows=n + len(xt), features=144, horizons=HORIZONS, labels='1=up,2=stationary,3=down', day_boundaries=bounds, exact_day_boundaries=exact, canonical_files=folds[9], prefix_counts=counts, note='CF1..8 used for prefix lengths only, not pooled. Final day never used for selection.'));return (x, y, xt, yt, bounds, exact)
def splits(n, bounds, exact, purge):
 if exact:pairs = [(bounds[k], bounds[k + 1]) for k in (5, 7, 8)]
 else:pairs = [(int(n * a), int(n * b)) for a, b in ((0.5, 0.65), (0.65, 0.8), (0.8, 1.0))]
 result = []
 for end, vend in pairs:
  tr = np.arange(max(0, end - purge));va = np.arange(end + purge, min(vend - purge, n))
  if min(len(tr), len(va)) < 25:raise ValueError('Too few rows after purging; reduce SETTINGS purge only for synthetic/small data')
  assert tr[-1] + purge < va[0];result.append((tr, va))
 return result
def preprocess(x, xe, kind):
 idx = np.arange(40 if kind == 'raw40' else 144);mean = x[:, idx].mean(0);sd = x[:, idx].std(0);good = sd > 1e-10;idx = idx[good];mean = mean[good];sd = sd[good];z = (x[:, idx] - mean) / sd;ze = (xe[:, idx] - mean) / sd;names = np.array(feature_names())[idx].tolist()
 if kind == 'basis':
  j = min(12, z.shape[1]);extras = [z[:, :j] ** 2];extrae = [ze[:, :j] ** 2];en = [s + '^2' for s in names[:j]]
  for a, b in ((0, 2), (1, 3), (4, 6), (5, 7)):
   if b < j:extras.append((z[:, a] * z[:, b])[:, None]);extrae.append((ze[:, a] * ze[:, b])[:, None]);en.append(names[a] + '*' + names[b])
  z = np.column_stack([z, *extras]);ze = np.column_stack([ze, *extrae]);names += en
 z = np.column_stack([np.ones(len(z)), z]);ze = np.column_stack([np.ones(len(ze)), ze]);names = ['intercept'] + names;_, r, piv = la.qr(z, mode='economic', pivoting=True, check_finite=False);diag = np.abs(np.diag(r));rank = int((diag > diag.max() * 1e-08).sum());keep = sorted(piv[:rank]);removed = [names[i] for i in range(len(names)) if i not in keep];return (np.ascontiguousarray(z[:, keep]), np.ascontiguousarray(ze[:, keep]), [names[i] for i in keep], removed)
_CUDA_SOLVER_INIT_LOCK = threading.Lock();_CUDA_SOLVERS_READY = set()
def initialize_cuda_solver(device):
 with _CUDA_SOLVER_INIT_LOCK:
  if device in _CUDA_SOLVERS_READY:return
  LOG.info('Initializing CUDA QR solver serially on %s', device)
  with torch.cuda.device(device), torch.no_grad():eye = torch.eye(8, dtype=torch.float64, device=device);q, r = torch.linalg.qr(eye, mode='reduced');torch.linalg.solve_triangular(r, q.T, upper=True);torch.cuda.synchronize(device)
  _CUDA_SOLVERS_READY.add(device)
class Algebra:
 def __init__(self, device):self.device = device
 def fit(self, x, y, w=None):
  if self.device == 'cpu':
   if w is not None:sw = np.sqrt(w);return la.lstsq(x * sw[:, None], y * sw[:, None], cond=1e-10, check_finite=False)[0]
   return la.lstsq(x, y, cond=1e-10, check_finite=False)[0]
  initialize_cuda_solver(self.device)
  with torch.cuda.device(self.device), torch.no_grad():
   a = torch.as_tensor(x, dtype=torch.float64, device=self.device);b = torch.as_tensor(y, dtype=torch.float64, device=self.device)
   if w is not None:s = torch.as_tensor(w, dtype=torch.float64, device=self.device).sqrt()[:, None];a = a * s;b = b * s
   q, r = torch.linalg.qr(a, mode='reduced');diagonal = torch.diagonal(r).abs()
   if diagonal.min() < diagonal.max() * 1e-10:LOG.warning('Subset lost numerical rank on %s; CPU SVD fallback', self.device);return Algebra('cpu').fit(x, y, w)
   coef = torch.linalg.solve_triangular(r, q.T @ b, upper=True)
   if not torch.isfinite(coef).all():raise ArithmeticError('Nonfinite GPU least-squares result')
   return coef.cpu().numpy()
def model_fit(x, y, method, engine):
 b = engine.fit(x, y);info = {}
 if method == 'wls':e = y - x @ b;v = np.column_stack([np.ones(len(x)), np.clip(x[:, 1:min(9, x.shape[1])], -8, 8)]);gamma = engine.fit(v, np.log(np.maximum(e ** 2, 1e-05)));logvar = v @ gamma;weights = np.exp(-np.clip(logvar.mean(1), -8, 8));weights = np.clip(weights, 0.05, 20);b = engine.fit(x, y, weights);info = {'weight_min': float(weights.min()), 'weight_max': float(weights.max())}
 if method == 'gls':e = y - x @ b;rho = np.clip((e[1:] * e[:-1]).sum() / max((e[:-1] ** 2).sum(), 1e-12), -0.95, 0.95);a = x[1:] - rho * x[:-1];t = y[1:] - rho * y[:-1];b = engine.fit(a, t);info = {'rho': float(rho), 'caveat': 'AR1 is row-order sensitivity; unknown instrument boundaries'}
 return (b, info)
def metrics(y, scores):pred = scores.argmax(1);pr, rc, fs, sup = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0);return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0), balanced_accuracy=float(np.mean(rc[np.asarray(sup) > 0])), mse_onehot=float(np.mean((np.eye(3)[y] - scores) ** 2)), outside_unit_fraction=float(np.mean((scores < 0) | (scores > 1))), per_class=dict(zip(CLASSES, [dict(precision=p, recall=r, f1=f, support=int(n)) for p, r, f, n in zip(pr, rc, fs, sup)])))
def eda(x, y, out, tag):
 names = feature_names();sd = x.std(0);nz = sd > 1e-10;z = (x[:, nz] - x[:, nz].mean(0)) / sd[nz];corr = z.T @ z / len(z);eig = np.maximum(la.eigvalsh(corr), 0);rows = []
 for j, n in enumerate(names):
  row = dict(feature=n, mean=x[:, j].mean(), std=sd[j], constant=not nz[j], min=x[:, j].min(), max=x[:, j].max())
  for h in range(5):row[f'corr_direction_h{HORIZONS[h]}'] = np.corrcoef(x[:, j], np.array([1, 0, -1])[y[:, h]])[0, 1] if nz[j] else np.nan
  rows.append(row)
 table(out / 'feature_summary.csv', rows);high = np.argwhere(np.triu(np.abs(corr) >= 0.95, 1));nms = np.array(names)[nz];table(out / 'high_correlation_pairs.csv', [dict(a=nms[i], b=nms[j], r=corr[i, j]) for i, j in high]);singular = eig < max(eig.max() * 1e-10, 1e-14);precision = la.pinvh(corr, rtol=1e-10);vif = np.diag(precision).copy()
 if singular.any():ev, vec = la.eigh(corr);alias = (vec[:, ev < eig.max() * 1e-10] ** 2).sum(1) > 1e-08;vif[alias] = np.inf
 table(out / 'collinearity.csv', [dict(feature=n, vif=v, aliased=np.isinf(v)) for n, v in zip(nms, vif)]);dump(out / 'collinearity.json', dict(nonconstant=int(nz.sum()), rank=int((~singular).sum()), constant=int((~nz).sum()), highly_correlated_pairs=len(high), eigenvalues=eig, condition_indices=np.sqrt(eig.max() / np.maximum(eig, 1e-30))));fig(out / 'correlation.png', f'{tag}: training predictor correlation', lambda a: heatmap(a, corr, cmap='coolwarm', vmin=-1, vmax=1));fig(out / 'condition_spectrum.png', 'Training correlation spectrum (near-zero = aliasing)', lambda a: a.semilogy(np.maximum(eig[::-1], 1e-18)));fig(out / 'target_balance.png', 'Training target proportions by supplied horizon', lambda a: [a.plot(HORIZONS, [(y[:, j] == c).mean() for j in range(5)], label=CLASSES[c], marker='o') for c in range(3)] + [a.legend(), a.set_xlabel('Horizon (events)')])
 for j, h in enumerate(HORIZONS):cm = confusion_matrix(y[:-1, j], y[1:, j], labels=[0, 1, 2]).astype(float);cm /= np.maximum(cm.sum(1, keepdims=True), 1);fig(out / f'target_transitions_h{h}.png', f'Adjacent-row target transitions h={h} (descriptive)', lambda a, c=cm: heatmap(a, c, CLASSES, CLASSES, vmin=0, vmax=1))
 fig(out / 'target_cross_horizon_agreement.png', 'Training class agreement across horizons', lambda a: heatmap(a, np.array([[(y[:, i] == y[:, j]).mean() for j in range(5)] for i in range(5)]), HORIZONS, HORIZONS, vmin=0, vmax=1));rolling = pd.DataFrame(np.eye(3)[y[:, 0]], columns=CLASSES).rolling(max(10, len(y) // 100)).mean();fig(out / 'target_drift_train.png', 'Training rolling class proportions, h=10', lambda a: [a.plot(rolling[c].to_numpy(), label=c) for c in CLASSES] + [a.legend()])
def inference(x, y, names, out, h, raw_design=None):
 n, p = x.shape;b = la.lstsq(x, y, cond=1e-10, check_finite=False)[0];fit = x @ b;e = y - fit;q, r = la.qr(x, mode='economic', check_finite=False);hat = np.clip((q * q).sum(1), 0, 1 - 1e-10);ri = la.solve_triangular(r, np.eye(p), check_finite=False);inv = ri @ ri.T;df = n - p;s2 = float(e @ e / df);denom = np.maximum(1 - hat, 1e-10);internal = e / np.sqrt(np.maximum(s2 * denom, 1e-20));deleted_s2 = np.maximum((e @ e - e ** 2 / denom) / max(df - 1, 1), 1e-20);external = e / np.sqrt(deleted_s2 * denom);cook = e ** 2 * hat / (p * max(s2, 1e-20) * denom ** 2);dffits = external * np.sqrt(hat / denom);covratio = (deleted_s2 / max(s2, 1e-20)) ** p / denom;se = np.sqrt(np.maximum(s2 * np.diag(inv), 1e-20));t = b / se;pv = 2 * stats.t.sf(np.abs(t), df);bon = stats.t.ppf(1 - 0.05 / (2 * p), df);sch = np.sqrt(p * stats.f.ppf(0.95, p, df));table(out / f'coefficients_h{h}.csv', [dict(feature=name, beta=bb, se_gaussian_IID_reference=ss, t_reference=tt, p_reference=pp, bonferroni_lo=bb - bon * ss, bonferroni_hi=bb + bon * ss, scheffe_lo=bb - sch * ss, scheffe_hi=bb + sch * ss) for name, bb, ss, tt, pp in zip(names, b, se, t, pv)]);relationships = []
 for j, name in enumerate(names):
  if name == 'intercept':continue
  marginal = np.corrcoef(x[:, j], y)[0, 1];relationships.append(dict(feature=name, marginal_correlation=marginal, partial_coefficient=b[j], sign_reversal=bool(marginal * b[j] < 0), parameterization='QR-selected independent columns; sign reversal is descriptive, not causation'))
 table(out / f'marginal_vs_partial_h{h}.csv', relationships);table(out / f'influence_h{h}.csv', dict(row=np.arange(n), fitted=fit, residual=e, leverage=hat, studentized_internal=internal, studentized_external=external, cook=cook, dffits=dffits, covratio=covratio, press_residual=e / denom));top = np.argsort(cook)[-min(200, n):];db = (e[top] / denom[top])[:, None] * (x[top] @ inv) / np.sqrt(deleted_s2[top, None] * np.maximum(np.diag(inv), 1e-20));pd.DataFrame(db, index=top, columns=names).to_csv(out / f'top_DFBETAS_h{h}.csv');raw = [i for i, nm in enumerate(names) if nm == 'intercept' or nm.startswith(('ask_', 'bid_'))];reduced = x[:, raw] if raw_design is None else raw_design;br, _, reduced_rank, _ = la.lstsq(reduced, y, cond=1e-10, check_finite=False);rssr = np.sum((y - reduced @ br) ** 2);dof = p - reduced_rank;fstat = max(0, (rssr - e @ e) / max(dof, 1) / max(s2, 1e-20));contrast = {}
 if 'ask_volume_L1' in names and 'bid_volume_L1' in names:c = np.zeros(p);c[names.index('ask_volume_L1')] = 1;c[names.index('bid_volume_L1')] = -1;value = c @ b;cs = np.sqrt(s2 * (c @ inv @ c));contrast = dict(estimate=value, se_reference=cs, scheffe_lo=value - sch * cs, scheffe_hi=value + sch * cs)
 dump(out / f'inference_h{h}.json', dict(response='direction score +1/0/-1; NOT return', n=n, p=p, df=df, RSS=e @ e, R2=1 - e @ e / max(np.sum((y - y.mean()) ** 2), 1e-20), PRESS=np.sum((e / denom) ** 2), engineered_F_reference=fstat, engineered_F_df=[dof, df], engineered_p_reference=stats.f.sf(fstat, dof, df) if dof else None, volume_contrast=contrast, warning=WARNINGS[0], max_leverage=hat.max(), sum_leverage=hat.sum(), influential_above_4_over_n=int((cook > 4 / n).sum())));stride = max(1, n // 6000);ix = np.arange(0, n, stride);fig(out / f'residual_fitted_h{h}.png', f'h={h}: residual vs fitted (direction score)', lambda a: a.scatter(fit[ix], e[ix], s=2, alpha=0.3));fig(out / f'scale_location_h{h}.png', f'h={h}: sqrt absolute studentized residual', lambda a: a.scatter(fit[ix], np.sqrt(np.abs(internal[ix])), s=2, alpha=0.3));fig(out / f'qq_h{h}.png', f'h={h}: normal QQ reference; discrete response violates Gaussian model', lambda a: stats.probplot(e[ix], plot=a));fig(out / f'cook_h{h}.png', f'h={h}: Cooks distance', lambda a: a.plot(cook));fig(out / f'leverage_h{h}.png', f'h={h}: leverage vs externally studentized residual', lambda a: a.scatter(hat[ix], external[ix], s=3));ac = [np.corrcoef(e[:-k], e[k:])[0, 1] for k in range(1, min(201, n // 5))];fig(out / f'residual_acf_h{h}.png', f'h={h}: row-order residual correlation (unknown stock boundaries)', lambda a: a.plot(np.arange(1, len(ac) + 1), ac));order = np.argsort(np.abs(b / se))[::-1];order = [i for i in order if names[i] != 'intercept'][:6]
 for j in order:fig(out / f'partial_residual_h{h}_{j}.png', f'h={h}: partial residual for {names[j]}', lambda a, j=j: a.scatter(x[ix, j], e[ix] + x[ix, j] * b[j], s=2, alpha=0.3))
 fig(out / f'coef_intervals_h{h}.png', f'h={h}: Bonferroni Gaussian IID reference intervals (not valid significance)', lambda a: [a.errorbar(b[order], np.arange(len(order)), xerr=bon * se[order], fmt='o'), a.set_yticks(np.arange(len(order)), [names[j] for j in order])]);mean_se = np.sqrt(np.maximum(s2 * hat, 0));table(out / f'mean_reference_bands_h{h}.csv', dict(row=ix, fitted=fit[ix], scheffe_mean_lo=fit[ix] - sch * mean_se[ix], scheffe_mean_hi=fit[ix] + sch * mean_se[ix]));return b
SPECS = [(kind, method) for kind in ('raw40', 'full144', 'basis') for method in ('ols', 'wls', 'gls')]
def targets(y):return np.eye(3)[y].reshape(len(y), 15)
def validate_view(x, y, bounds, exact, purge, devices, out):
 records = []
 for fold, (tr, va) in enumerate(splits(len(x), bounds, exact, purge)):
  for kind in ('raw40', 'full144', 'basis'):
   a, v, names, removed = preprocess(x[tr], x[va], kind)
   for mi, method in enumerate(('ols', 'wls', 'gls')):
    engine = Algebra(devices[mi % len(devices)]);b, info = model_fit(a, targets(y[tr]), method, engine);score = (v @ b).reshape(len(v), 5, 3)
    for j, h in enumerate(HORIZONS):records.append(dict(fold=fold, horizon=h, kind=kind, method=method, n_train=len(tr), n_valid=len(va), rank=a.shape[1], **{k: v for k, v in metrics(y[va, j], score[:, j]).items() if k != 'per_class'}))
    LOG.info('Validation fold=%d %s/%s complete on %s', fold, kind, method, engine.device)
  table(out / 'validation_scores.csv', records)
 df = pd.DataFrame(records);agg = df.groupby(['horizon', 'kind', 'method'], as_index=False).macro_f1.mean();winners = {int(h): g.sort_values(['macro_f1', 'kind', 'method'], ascending=[False, True, True]).iloc[0][['kind', 'method']].to_dict() for h, g in agg.groupby('horizon')};dump(out / 'frozen_selection.json', dict(winners=winners, criterion='mean purged chronological validation macro-F1', purge=purge, protocols='3 expanding windows within days 1..9; day10 not inspected during selection', specs=SPECS));table(out / 'validation_mean.csv', agg);pivot = agg.pivot(index=['kind', 'method'], columns='horizon', values='macro_f1');fig(out / 'validation_comparison.png', 'Mean purged chronological validation macro-F1', lambda a: heatmap(a, pivot.to_numpy(), HORIZONS, ['/'.join(i) for i in pivot.index], vmin=0, vmax=1));return winners
def final_view(x, y, xt, yt, winners, purge, devices, out, diagnostics=True):
 xr = x[:-purge] if purge else x;yr = y[:-purge] if purge else y;xe = xt[purge:len(xt) - purge] if purge else xt;ye = yt[purge:len(yt) - purge] if purge else yt
 if len(xe) < 25:raise ValueError('Held-out test too short after endpoint purge')
 results = [];frozen_predictions = {}
 for ki, kind in enumerate(('raw40', 'full144', 'basis')):
  a, v, names, removed = preprocess(xr, xe, kind);engine = Algebra(devices[ki % len(devices)]);dump(out / f'column_space_{kind}.json', dict(kept=names, removed_as_constant_or_aliased=removed, warning='Dropped-column coefficients are not identifiable; retained coefficients use this QR parameterization.'));drift = (xe.mean(0) - xr.mean(0)) / np.maximum(xr.std(0), 1e-10);table(out / 'heldout_standardized_drift.csv', [dict(feature=n, z_mean_shift=d) for n, d in zip(feature_names(), drift)])
  for method in ('ols', 'wls', 'gls'):
   b, info = model_fit(a, targets(yr), method, engine);s = (v @ b).reshape(len(v), 5, 3);np.savez_compressed(out / f'model_{kind}_{method}.npz', coef=b, names=names, train_feature_mean=xr.mean(0), train_feature_std=xr.std(0), info=json.dumps(info), note='Use predict_saved(model_path,new_144_column_matrix) from this script')
   for j, h in enumerate(HORIZONS):
    m = metrics(ye[:, j], s[:, j]);selected = winners[h] == dict(kind=kind, method=method);results.append(dict(kind=kind, method=method, horizon=h, selected_on_validation=selected, **m));cm = confusion_matrix(ye[:, j], s[:, j].argmax(1), labels=[0, 1, 2]);fig(out / f'confusion_{kind}_{method}_h{h}.png', f'{kind}/{method} h={h} final holdout; rows=true, cols=pred', lambda a, c=cm: heatmap(a, c, CLASSES, CLASSES))
    if selected:frozen_predictions[h] = (s[:, j], ye[:, j]);table(out / f'selected_predictions_h{h}.csv', dict(row=np.arange(purge, purge + len(ye)), truth_code=ye[:, j] + 1, pred_code=s[:, j].argmax(1) + 1, up_score=s[:, j, 0], stationary_score=s[:, j, 1], down_score=s[:, j, 2]))
   if method == 'ols' and kind == 'full144' and diagnostics:
    direction = np.array([1.0, 0.0, -1.0])[yr];raw_a, _, _, _ = preprocess(xr, xe, 'raw40')
    for j, h in enumerate(HORIZONS):inference(a, direction[:, j], names, out, h, raw_a)
   LOG.info('Final fixed model %s/%s evaluated', kind, method)
 for j, h in enumerate(HORIZONS):maj = np.bincount(yr[:, j], minlength=3).argmax();sc = np.tile(np.eye(3)[maj], (len(ye), 1));results.append(dict(kind='baseline', method='training_majority', horizon=h, selected_on_validation=False, **metrics(ye[:, j], sc)))
 dump(out / 'final_metrics.json', results);table(out / 'final_metrics.csv', [{k: v for k, v in r.items() if k != 'per_class'} for r in results])
 for h, (score, truth) in frozen_predictions.items():
  j = HORIZONS.index(h);bins = np.unique(np.quantile(xr[:, 1], np.linspace(0, 1, 11)));group = np.clip(np.digitize(xe[:, 1], bins) - 1, 0, len(bins) - 2);rows = []
  for g in range(len(bins) - 1):
   mask = group == g
   if mask.sum():rows.append(dict(bin=g, n=int(mask.sum()), up_rate=(truth[mask] == 0).mean(), down_rate=(truth[mask] == 2).mean(), macro_f1=metrics(truth[mask], score[mask])['macro_f1']))
  table(out / f'heldout_volume_bins_h{h}.csv', rows);fig(out / f'heldout_volume_response_h{h}.png', f'h={h}: heldout outcomes by training-defined ask-volume bins', lambda a, r=rows: a.plot([q['bin'] for q in r], [q['up_rate'] - q['down_rate'] for q in r], marker='o'));table(out / f'nonoverlap_offsets_h{h}.csv', [dict(offset=o, n=len(truth[o::h]), **{k: v for k, v in metrics(truth[o::h], score[o::h]).items() if k != 'per_class'}) for o in range(min(h, len(truth)))])
 return results
def family_analysis(raw, labels, out):
 sd = raw.std(0);z = (raw - raw.mean(0)) / np.maximum(sd, 1e-10);z = np.column_stack([np.ones(len(z)), z]);target = np.array([1.0, 0.0, -1.0])[labels];full, _, rank, _ = la.lstsq(z, target, cond=1e-08, check_finite=False);rss = np.sum((target - z @ full) ** 2, axis=0);rows = [];families = {'raw_prices': [j for j in range(40) if j % 4 in (0, 2)], 'raw_volumes': [j for j in range(40) if j % 4 in (1, 3)], 'engineered': list(range(40, 144))}
 for name, removed in families.items():
  keep = [0] + [j + 1 for j in range(144) if j not in removed];reduced = z[:, keep];beta, _, r, _ = la.lstsq(reduced, target, cond=1e-08, check_finite=False);rss0 = np.sum((target - reduced @ beta) ** 2, axis=0);d = rank - r
  for j, h in enumerate(HORIZONS):partial = max(0, 1 - rss[j] / max(rss0[j], 1e-20));f = max(0, (rss0[j] - rss[j]) / max(d, 1) / (rss[j] / (len(z) - rank)));rows.append(dict(family=name, horizon=h, rank_lost=d, partial_R2=partial, F_Gaussian_IID_reference=f if d else None, p_Gaussian_IID_reference=stats.f.sf(f, d, len(z) - rank) if d else None, note='Zero rank loss means family is algebraically redundant given remaining columns; all tests are descriptive reference calculations'))
 table(out / 'family_incremental_explanation.csv', rows);fig(out / 'family_partial_R2.png', 'Training partial R squared: contribution beyond remaining families', lambda a: [a.plot(HORIZONS, [r['partial_R2'] for r in rows if r['family'] == f], label=f, marker='o') for f in families] + [a.legend(), a.set_xlabel('Horizon')])
class ResourceMonitor:
 def __init__(self, out):self.out = out;self.stop = threading.Event();self.thread = None
 def start(self):
  def sample():
   while not self.stop.is_set():
    row = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if psutil:row.update(rss_bytes=psutil.Process().memory_info().rss, available_ram_bytes=psutil.virtual_memory().available, cpu_percent=psutil.cpu_percent())
    row['gpu'] = [dict(device=i, allocated_bytes=torch.cuda.memory_allocated(i), reserved_bytes=torch.cuda.memory_reserved(i), peak_allocated_bytes=torch.cuda.max_memory_allocated(i)) for i in range(torch.cuda.device_count())]
    with (self.out / 'resources.jsonl').open('a') as f:f.write(json.dumps(row) + '\n')
    self.stop.wait(15)
  self.thread = threading.Thread(target=sample, daemon=True);self.thread.start()
 def close(self):
  self.stop.set()
  if self.thread:self.thread.join(timeout=3)
def predict_saved(model_path, raw_features):
 with np.load(model_path) as model:
  raw = np.asarray(raw_features, dtype=float)
  if raw.ndim != 2 or raw.shape[1] != 144:raise ValueError('Expected n by 144 features in original FI order')
  sd = np.maximum(model['train_feature_std'], 1e-10);zz = (raw - model['train_feature_mean']) / sd;values = {name: zz[:, j] for j, name in enumerate(feature_names())};cols = []
  for name in model['names'].tolist():
   if name == 'intercept':cols.append(np.ones(len(raw)))
   elif name.endswith('^2'):cols.append(values[name[:-2]] ** 2)
   elif '*' in name:a, b = name.split('*');cols.append(values[a] * values[b])
   else:cols.append(values[name])
  return (np.column_stack(cols) @ model['coef']).reshape(len(raw), 5, 3)
def gpu_test(output=None):
 if not torch.cuda.is_available():raise RuntimeError('CUDA is not accessible to this Python process. GPU test cannot run; use a CUDA-enabled local/Kaggle session.')
 rng = np.random.default_rng(91);a = np.column_stack([np.ones(8192), rng.normal(size=(8192, 47))]);b = rng.normal(size=(8192, 15));rows = []
 def check(device):
  torch.cuda.reset_peak_memory_stats(device)
  for method in ('ols', 'wls', 'gls'):
   ref, _ = model_fit(a, b, method, Algebra('cpu'));start = time.monotonic();got, _ = model_fit(a, b, method, Algebra(device));error = float(np.max(np.abs(got - ref)))
   if not np.allclose(got, ref, atol=1e-08, rtol=1e-07):raise AssertionError(f'{device}/{method}: CPU/GPU mismatch {error}')
   rows.append(dict(device=device, method=method, max_abs_error=error, seconds=time.monotonic() - start, peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
 devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
 for device in devices:initialize_cuda_solver(device)
 with cf.ThreadPoolExecutor(max_workers=len(devices)) as pool:list(pool.map(check, devices))
 result = dict(status='passed', devices=devices, tests=rows, two_device_concurrency_tested=len(devices) >= 2, note='Uses float64 QR, triangular solve, feasible WLS and GLS; compares against CPU SVD. Synthetic only.')
 if output is not None:dump(output, result)
 print('GPU_TEST_PASSED ' + json.dumps(result), flush=True);return result
def hardware():gpus = [dict(index=i, name=torch.cuda.get_device_name(i), vram_bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())];return dict(cpu_count=os.cpu_count(), ram_bytes=psutil.virtual_memory().total if psutil else None, gpus=gpus, torch_version=torch.__version__, numpy_version=np.__version__, note='Uses actual visible devices. Two P100 GPUs are not assumed. Float64 QR; no n-by-n matrix. Allocation is bounded by data and useful work, not nominal capacity.')
def deletion_study(x, y, devices, out, deadline, seed, max_rounds):
 if deadline <= time.monotonic() or max_rounds == 0:return
 designs = {};t = targets(y);baseline = {};moments = {}
 for kind in ('raw40', 'full144', 'basis'):z, _, names, _ = preprocess(x, x[:1], kind);designs[kind] = (z, names)
 for kind, method in SPECS:z, names = designs[kind];key = kind + '_' + method;coef, _ = model_fit(z, t, method, Algebra(devices[len(baseline) % len(devices)]));baseline[key] = coef;moments[key] = dict(n=0, mean=np.zeros_like(coef), m2=np.zeros_like(coef), sign=np.zeros_like(coef))
 anchor = np.linspace(0, len(x) - 1, min(10000, len(x)), dtype=int)
 def job(device, round_id):kind, method = SPECS[round_id % len(SPECS)];key = kind + '_' + method;z, names = designs[kind];base = baseline[key];rng = np.random.default_rng(seed + round_id);frac = (0.01, 0.05, 0.1, 0.2)[round_id // len(SPECS) % 4];length = max(1, int(len(z) * frac));start = int(rng.integers(0, len(z) - length + 1));keep = np.r_[np.arange(start), np.arange(start + length, len(z))];started = time.monotonic();coef, _ = model_fit(z[keep], t[keep], method, Algebra(device));delta = coef - base;np.savez_compressed(out / f'deletion_{round_id:06d}.npz', coef=coef, start=start, length=length, device=device, specification=key);score = (z[anchor] @ coef).reshape(len(anchor), 5, 3);ref = (z[anchor] @ base).reshape(len(anchor), 5, 3);row = dict(round=round_id, specification=key, device=device, start=start, length=length, deleted_fraction=frac, relative_coef_change=float(la.norm(delta) / max(la.norm(base), 1e-12)), max_coef_change=float(np.abs(delta).max()), seconds=time.monotonic() - started, training_anchor_class_disagreement=float(np.mean(score.argmax(2) != ref.argmax(2))));return (row, key, coef)
 def save_moments():
  records = []
  for kind, method in SPECS:
   key = kind + '_' + method;m = moments[key];names = designs[kind][1]
   if not m['n']:continue
   std = np.sqrt(m['m2'] / max(m['n'] - 1, 1));agreement = m['sign'] / m['n'];np.savez_compressed(out / f'deletion_moments_{key}.npz', n=m['n'], mean=m['mean'], std=std, sign_agreement=agreement, names=names)
   for i, name in enumerate(names):records.append(dict(specification=key, feature=name, refits=m['n'], **{f'h{h}_{CLASSES[k]}_sign_agreement': agreement[i, 3 * j + k] for j, h in enumerate(HORIZONS) for k in range(3)}))
  table(out / 'coefficient_sign_stability.csv', records)
 rows = [];rid = 0;last = time.monotonic()
 with cf.ThreadPoolExecutor(max_workers=len(devices)) as pool:
  while time.monotonic() < deadline and (max_rounds < 0 or rid < max_rounds):
   count = len(devices) if max_rounds < 0 else min(len(devices), max_rounds - rid);futures = [pool.submit(job, dev, rid + i) for i, dev in enumerate(devices[:count])]
   for future in futures:row, key, coef = future.result();rows.append(row);m = moments[key];m['n'] += 1;delta = coef - m['mean'];m['mean'] += delta / m['n'];m['m2'] += delta * (coef - m['mean']);m['sign'] += np.sign(coef) == np.sign(baseline[key]);pd.DataFrame([row]).to_csv(out / 'block_deletion_scores.csv', mode='a', header=not (out / 'block_deletion_scores.csv').exists(), index=False)
   rid += count;dump(out / 'progress.json', dict(stage='nine-specification block-deletion study', replicates=rid, per_specification={k: m['n'] for k, m in moments.items()}, seconds_to_compute_deadline=max(0, deadline - time.monotonic())))
   if time.monotonic() - last > 30 or rid == count:LOG.info('Block deletion: %d refits; %.1f minutes left', rid, max(0, deadline - time.monotonic()) / 60);save_moments();last = time.monotonic()
   if rid > 256:
    for q in out.glob('deletion_*.npz'):
     if q.stem.split('_')[-1].isdigit() and int(q.stem.split('_')[-1]) < rid - 256:q.unlink()
 save_moments()
 if rows:
  fig(out / 'block_deletion_sensitivity.png', 'W11: coefficient sensitivity by omitted block size and specification', lambda a: [a.scatter([r['deleted_fraction'] for r in rows if r['specification'] == key], [r['relative_coef_change'] for r in rows if r['specification'] == key], s=6, alpha=0.3, label=key) for key in moments] + [a.legend(fontsize=7)])
  for key, m in moments.items():
   if m['n']:fig(out / f'coefficient_sign_stability_{key}.png', f"{key}: sign agreement across ALL {m['n']} block-deletion refits", lambda a, m=m: heatmap(a, m['sign'] / m['n'], [f'{h}:{c}' for h in HORIZONS for c in CLASSES], vmin=0, vmax=1))
 dump(out / 'deletion_summary.json', dict(total_refits=len(rows), coefficient_matrices_retained=min(256, len(rows)), per_specification={k: m['n'] for k, m in moments.items()}, evidence='Online coefficient moments and signs include every refit, not only retained raw matrices', scope='Training data only; sensitivity, NOT confidence or independent observations', seed=seed))
def report(out, summary):
 figures = sorted(out.rglob('*.png'));outputs = sorted(out.rglob('*.csv'));claims = []
 for v in summary.get('views', []):
  final = out / v['name'] / 'final_metrics.csv'
  if final.exists():
   df = pd.read_csv(final);sel = df[df.selected_on_validation == True];base = df[df.kind == 'baseline']
   for _, r in sel.iterrows():b = base[base.horizon == r.horizon].iloc[0];claims.append(f"{v['name']}, h={int(r.horizon)}: validation-selected {r.kind}/{r.method}, heldout macro-F1 {r.macro_f1:.4f}; training-majority {b.macro_f1:.4f}; difference {r.macro_f1 - b.macro_f1:+.4f}. This is a descriptive heldout comparison.")
 body = ['<!doctype html><html><head><meta charset="utf-8"><title>FI-2010 regression laboratory</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;color:#172033}img{max-width:100%;border:1px solid #ddd}pre{white-space:pre-wrap}details{margin:15px 0}a{color:#165abd}</style></head><body>', '<h1>FI-2010: Regression Techniques laboratory</h1>', '<p>Results are computed at runtime. Final-day outcomes do not choose models. Class-score fits are not return forecasts.</p>', '<h2>Read before interpreting</h2><ul>' + ''.join(('<li>' + html.escape(s) + '</li>' for s in WARNINGS)) + '</ul>', '<h2>Course map</h2><ul>' + ''.join(('<li>' + html.escape(k + ': ' + v) + '</li>' for k, v in COURSE.items())) + '</ul>', '<h2>Measured findings</h2><ul>' + ''.join(('<li>' + html.escape(s) + '</li>' for s in claims)) + '</ul>', '<h2>Run manifest</h2><pre>' + html.escape(json.dumps(summary, indent=2, default=str)) + '</pre>', '<h2>Tables</h2><ul>' + ''.join((f'<li><a href="{q.relative_to(out).as_posix()}">{html.escape(str(q.relative_to(out)))}</a></li>' for q in outputs)) + '</ul>', '<h2>Figure gallery</h2>']
 for q in figures:rel = q.relative_to(out).as_posix();body.append(f'<details><summary>{html.escape(rel)}</summary><img loading="lazy" src="{rel}"></details>')
 body += ['<h2>Sources</h2><p>Local course notes W01-W12 and model.pdf. Dataset: <a href="https://www.kaggle.com/datasets/alpha000x/fi-2010">alpha000x/fi-2010</a>. Benchmark paper: <a href="https://arxiv.org/abs/1705.03233">Ntakaris et al.</a>.</p></body></html>'];(out / 'report.html').write_text('\n'.join(body));(out / 'FINDINGS.md').write_text('# Runtime findings\n\n' + '\n\n'.join(claims or ['No final evaluation completed. Read run_summary.json.']) + '\n\n' + '\n\n'.join(WARNINGS));dump(out / 'artifact_counts.json', dict(figures=len(figures), csv_tables=len(outputs)))
def default_config(**overrides):values = dict(SETTINGS);values.update(overrides);values['input'] = Path(values['input']);values['output'] = Path(values['output']);return SimpleNamespace(**values)
def test_cuda_initialization():
 from unittest.mock import patch;from contextlib import contextmanager;local = threading.local();events = [];state = {'active': 0, 'peak': 0};fail = {'cuda:2'};saved = set(_CUDA_SOLVERS_READY)
 @contextmanager
 def device_context(device):local.device = device;yield
 def qr(a, mode):
  state['active'] += 1;state['peak'] = max(state['peak'], state['active'])
  try:
   time.sleep(0.01)
   if state['active'] > 1:raise RuntimeError('lazy wrapper should be called at most once')
   if local.device in fail:fail.remove(local.device);raise RuntimeError('injected initialization failure')
   events.append(('qr', local.device));return (a, a)
  finally:state['active'] -= 1
 def solve(r, b, upper):events.append(('solve', local.device));return b
 def sync(device):events.append(('sync', device))
 try:
  _CUDA_SOLVERS_READY.clear()
  with patch.object(torch.cuda, 'device', side_effect=device_context), patch.object(torch, 'eye', side_effect=lambda n, **kw: np.eye(n)), patch.object(torch.linalg, 'qr', side_effect=qr), patch.object(torch.linalg, 'solve_triangular', side_effect=solve), patch.object(torch.cuda, 'synchronize', side_effect=sync):
   with cf.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(initialize_cuda_solver, ['cuda:0', 'cuda:1'] * 8))
   assert state['peak'] == 1
   for device in ('cuda:0', 'cuda:1'):assert [e[0] for e in events if e[1] == device] == ['qr', 'solve', 'sync']
   before = list(events);initialize_cuda_solver('cuda:0');assert events == before
   try:initialize_cuda_solver('cuda:2')
   except RuntimeError as error:assert str(error) == 'injected initialization failure'
   else:raise AssertionError('Initialization failure was swallowed')
   assert 'cuda:2' not in _CUDA_SOLVERS_READY;initialize_cuda_solver('cuda:2');assert 'cuda:2' in _CUDA_SOLVERS_READY
 finally:_CUDA_SOLVERS_READY.clear();_CUDA_SOLVERS_READY.update(saved)
 print('CUDA_INIT_TEST_PASSED')
def run(args):
 if args.hours < 0 or args.purge < 0:raise ValueError('hours and purge must be nonnegative')
 out = args.output
 if out.exists() and any(out.iterdir()):
  stem = out.name + '_' + time.strftime('%Y%m%d_%H%M%S');candidate = out.with_name(stem);suffix = 1
  while candidate.exists():candidate = out.with_name(stem + '_' + str(suffix));suffix += 1
  out = candidate;args.output = out
 out.mkdir(parents=True, exist_ok=True);logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(out / 'run.log')], force=True);started = time.monotonic();deadline = started + args.hours * 3600;compute_end = started + args.hours * 3600 * 0.98;torch.set_num_threads(min(4, os.cpu_count() or 1));np.random.seed(args.seed);hw = hardware();devices = [f'cuda:{i}' for i in range(len(hw['gpus']))] or ['cpu']
 if hw['gpus']:gpu_test(out / 'gpu_test.json')
 dump(out / 'hardware.json', hw);dump(out / 'config.json', vars(args));dump(out / 'course_map.json', COURSE);LOG.info('Visible compute devices: %s. No GPU count/model assumed.', devices);groups = discover(args.input, out);keys = sorted(groups, key=lambda k: (k != ('NoAuction', 'Zscore'), k))
 if args.primary_only:keys = keys[:1]
 summary = dict(status='running', views=[], warnings=WARNINGS, requested_hours=args.hours, hardware=hw, scope='All available complete CF9 views' if not args.primary_only else 'Explicit primary-only reduced scope', final_evaluation='All fixed model results reported; validation chooses one per horizon; no final-test reranking');primary = None;signatures = {};monitor = ResourceMonitor(out);monitor.start()
 try:
  for key in keys:
   name = '_'.join(key);vout = out / name;vout.mkdir(exist_ok=True);LOG.info('Loading full ten-day representation %s', name);x, y, xt, yt, bounds, exact = load_view(groups[key], vout);sig = (len(x), len(xt), hashlib.sha256(y.tobytes() + yt.tobytes()).hexdigest())
   if key[0] in signatures and signatures[key[0]] != sig:raise ValueError('Label/order mismatch across normalization views; cannot compare as the same observations')
   signatures[key[0]] = sig;eda(x, y, vout, name)
   if key == keys[0]:family_analysis(x, y, vout)
   win = validate_view(x, y, bounds, exact, args.purge, devices, vout);final_view(x, y, xt, yt, win, args.purge, devices, vout, diagnostics=key == keys[0]);summary['views'].append(dict(name=name, train_rows=len(x), test_rows=len(xt), all_rows=len(x) + len(xt), exact_days=exact))
   if key == keys[0]:primary = (groups[key], vout)
   dump(out / 'run_summary.json', summary);report(out, summary);del x, y, xt, yt
  if primary and args.hours > 0 and (time.monotonic() < compute_end) and (args.max_deletions != 0):folds, vout = primary;x, y, xt, yt, bounds, exact = load_view(folds, vout);del xt, yt;xr = x[:-args.purge] if args.purge else x;yr = y[:-args.purge] if args.purge else y;deletion_study(xr, yr, devices, vout, compute_end, args.seed, args.max_deletions)
  summary['status'] = 'complete';summary['elapsed_seconds'] = time.monotonic() - started;summary['budget_overrun'] = args.hours > 0 and time.monotonic() > deadline;summary['budget_note'] = 'Finite required analysis always completes; optional deletion refits stop at 98% of target budget. No deliberate idle time.'
 except Exception as e:summary['status'] = 'failed';summary['error'] = str(e);summary['traceback'] = traceback.format_exc();LOG.exception('Run failed; partial artifacts retained');raise
 finally:monitor.close();summary['elapsed_seconds'] = time.monotonic() - started;dump(out / 'run_summary.json', summary);report(out, summary)
 LOG.info('Finished. Open %s', out / 'report.html');return summary
def self_test():
 with tempfile.TemporaryDirectory(prefix='fi2010_test_') as d:
  root = Path(d);folder = root / 'input' / 'NoAuction' / '1.NoAuction_Zscore';folder.mkdir(parents=True);rng = np.random.default_rng(18);n = 1200;x = rng.normal(size=(n, 144));x[:, 140:] = 0;x[:, 130] = x[:, 0] * 2;x[:, 131] = x[:, 1] + x[:, 2];signal = x[:, 0] + 0.3 * x[:, 1] + rng.normal(size=n) * 0.6;y = np.column_stack([np.where(signal + j * 0.08 > 0.5, 1, np.where(signal + j * 0.08 < -0.5, 3, 2)) for j in range(5)]);a = np.column_stack([x, y]);boundary = 1000
  for k in range(1, 10):end = 100 * k + 100 if k < 9 else boundary;np.savetxt(folder / f'Train_Dst_NoAuction_ZScore_CF_{k}.txt', a[:end].T);np.savetxt(folder / f'Test_Dst_NoAuction_ZScore_CF_{k}.txt', a[end:boundary if k < 9 else n].T)
  got = discover(root / 'input', root / 'scan');assert len(got) == 1;xx, yy, xe, ye, b, exact = load_view(next(iter(got.values())), root / 'audit');assert exact and len(xx) + len(xe) == n and np.array_equal(yy, y[:boundary] - 1);z, ze, names, removed = preprocess(xx, xe, 'full144');assert z.shape[1] < 141 and removed;beta = Algebra('cpu').fit(z, targets(yy));assert np.max(np.abs(z.T @ (targets(yy) - z @ beta))) < 1e-07
  for tr, va in splits(len(xx), b, True, 2):assert tr[-1] + 2 < va[0]
  z2, _, names2, _ = preprocess(xx, xe * 10000, 'basis');z1, _, names1, _ = preprocess(xx, xe, 'basis');assert names1 == names2 and np.allclose(z1, z2);bad = folder / 'bad.npy';np.save(bad, np.column_stack([xx, np.zeros((len(xx), 5))]))
  try:read_matrix(bad)
  except ValueError:pass
  else:raise AssertionError('Negative label validation control failed')
  bad.unlink();args = default_config(input=root / 'input', output=root / 'results', hours=0, purge=2, max_deletions=0);summary = run(args);out = args.output;assert summary['status'] == 'complete';v = out / 'NoAuction_Zscore';assert len(json.loads((v / 'final_metrics.json').read_text())) == 50;assert len(list(out.rglob('*.png'))) >= 70;assert (out / 'report.html').exists() and (v / 'influence_h100.csv').exists()
  for kind in ('raw40', 'full144', 'basis'):
   aa, vv, _, _ = preprocess(xx[:-2], xe[2:-2], kind)
   with np.load(v / f'model_{kind}_ols.npz') as m:expected = (vv @ m['coef']).reshape(len(vv), 5, 3)
   assert np.allclose(predict_saved(v / f'model_{kind}_ols.npz', xe[2:-2]), expected)
  import shutil;qa = Path('/tmp/fi2010_visual_qa');qa.mkdir(exist_ok=True)
  for name in ('coef_intervals_h10.png', 'residual_fitted_h10.png', 'validation_comparison.png'):shutil.copy(v / name, qa / name)
  inf = json.loads((v / 'inference_h10.json').read_text());assert abs(inf['sum_leverage'] - inf['p']) < 1e-07;ydir = np.array([1.0, 0.0, -1.0])[yy[:, 0]];coef = la.lstsq(z, ydir)[0];gram = la.pinvh(z.T @ z);i = 17;e = ydir[i] - z[i] @ coef;h = z[i] @ gram @ z[i];actual = la.lstsq(np.delete(z, i, 0), np.delete(ydir, i))[0];formula = coef - gram @ z[i] * e / (1 - h);assert np.allclose(formula, actual, atol=1e-08);deletion_study(xx, yy, ['cpu'], v, time.monotonic() + 30, 17, 9);assert json.loads((v / 'deletion_summary.json').read_text())['total_refits'] == 9
  for i in range(torch.cuda.device_count()):actual = Algebra(f'cuda:{i}').fit(z, targets(yy));assert np.allclose(actual, beta, atol=1e-07)
 if torch.cuda.is_available():gpu_test()
 print('SELF_TEST_PASSED', flush=True)
def verify_source():import ast;source = Path(__file__).read_text();tree = ast.parse(source);prohibited = {'requests', 'urllib', 'httpx', 'subprocess'};imported = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)};assert not imported & prohibited;assert not any((isinstance(n, ast.Attribute) and n.attr in ('backward', 'DataParallel') for n in ast.walk(tree)));a = default_config();assert a.hours == 8 and a.purge >= max(HORIZONS);assert 'arg' + 'parse' not in imported and 'run(default_config())' in source;assert {'self_test', 'deletion_study', 'inference', 'discover', 'validate_view', 'final_view'} <= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)};print('SOURCE_VERIFIED')
if __name__ == '__main__':run(default_config())
