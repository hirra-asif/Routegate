"""One-hidden-layer outcome heads for the fixed-edit selector. CPU only, saved responses only.

Everything except the model family is imported from experiments/outcome_selector_v1:
the target definitions, the fold arithmetic, the threshold search and the scoring.
"""
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from selector_core import binary_loss, choose_rules, apply_rule, id_digest, targets


METHOD = 'nn_outcome_selector'


def fit_nn_head(x, y, alpha, spec):
    """Fit one binary head as the mean of len(spec['seeds']) identically trained networks.

    Unknown targets are dropped, never imputed. The scaler sees only labelled training rows.
    """
    known = np.isfinite(y)
    values = y[known]
    result = {'alpha': float(alpha), 'known_training_rows': int(known.sum()),
              'positives': int(values.sum()), 'unknown_training_rows': int((~known).sum())}
    if len(np.unique(values)) < 2:
        result.update(kind='constant', value=float((values.sum() + 1) / (len(values) + 2)))
        return result
    scaler = StandardScaler().fit(x[known])
    design_matrix = scaler.transform(x[known])
    labels = values.astype(int)
    nets, seed_fits = [], []
    for seed in spec['seeds']:
        with warnings.catch_warnings():
            # Non-convergence is recorded and reported, not silently accepted and not fatal.
            warnings.simplefilter('ignore', ConvergenceWarning)
            model = MLPClassifier(hidden_layer_sizes=tuple(spec['hidden_layer_sizes']),
                                  activation=spec['activation'], solver=spec['solver'],
                                  alpha=float(alpha), max_iter=int(spec['max_iter']),
                                  tol=float(spec['tol']), random_state=int(seed))
            model.fit(design_matrix, labels)
        if list(model.classes_) != [0, 1]:
            raise ValueError('Unexpected head class order')
        if len(model.coefs_) != 2 or model.coefs_[0].shape[1] != spec['hidden_layer_sizes'][0]:
            raise ValueError('Unexpected network shape')
        nets.append((np.asarray(model.coefs_[0], dtype=np.float64), np.asarray(model.intercepts_[0], dtype=np.float64),
                     np.asarray(model.coefs_[1], dtype=np.float64)[:, 0], float(model.intercepts_[1][0])))
        seed_fits.append({'seed': int(seed), 'iterations': int(model.n_iter_),
                          'final_training_loss': float(model.loss_),
                          'converged': bool(int(model.n_iter_) < int(spec['max_iter']))})
    result.update(kind='mlp', hidden_units=int(spec['hidden_layer_sizes'][0]), seed_fits=seed_fits,
                  seeds_converged=int(sum(f['converged'] for f in seed_fits)), seeds_fitted=len(seed_fits),
                  parameters_per_network=int(x.shape[1] * spec['hidden_layer_sizes'][0] + 2 * spec['hidden_layer_sizes'][0] + 1))
    result['_arrays'] = {'mean': scaler.mean_, 'scale': scaler.scale_, 'nets': nets}
    return result


def predict_nn_head(model, x):
    """Seed-averaged probability. The same averaging is used in validation and in evaluation."""
    if model['kind'] == 'constant':
        return np.full(len(x), model['value'], dtype=float)
    arrays = model['_arrays']
    z = (x - arrays['mean']) / arrays['scale']
    total = np.zeros(len(x), dtype=float)
    for first_weight, first_bias, second_weight, second_bias in arrays['nets']:
        hidden = np.maximum(z @ first_weight + first_bias, 0.0)
        total += expit(hidden @ second_weight + second_bias)
    return total / len(arrays['nets'])


def strip(model):
    return {k: v for k, v in model.items() if k != '_arrays'}


def head_weights(model, prefix):
    if model['kind'] == 'constant':
        return {}
    arrays = model['_arrays']
    out = {prefix + '_mean': arrays['mean'].astype(np.float32), prefix + '_scale': arrays['scale'].astype(np.float32)}
    for i, (w1, b1, w2, b2) in enumerate(arrays['nets']):
        out[prefix + '_seed' + str(i) + '_w1'] = w1.astype(np.float32)
        out[prefix + '_seed' + str(i) + '_b1'] = b1.astype(np.float32)
        out[prefix + '_seed' + str(i) + '_w2'] = w2.astype(np.float32)
        out[prefix + '_seed' + str(i) + '_b2'] = np.float32(b2)
    return out


def fit_outer_nn(x, rows, outer, design):
    """Mirror of selector_core.fit_outer with the network family substituted for logistic regression.

    Outer-fold labels and features never touch scaling, network weights, alpha or thresholds.
    """
    spec = design['nn']
    folds = np.asarray([r['fold'] for r in rows])
    train_idx, test_idx = np.where(folds != outer)[0], np.where(folds == outer)[0]
    train_rows = [rows[i] for i in train_idx]
    train_x = x[train_idx]
    y = targets(train_rows, 'outcome_selector')  # identical joint-outcome targets; outer targets are never built here
    local_folds = folds[train_idx]
    scores = np.empty((len(train_idx), 3))
    final_models, head_history = [], []
    for head in range(3):
        candidates = []
        for alpha in spec['alpha_values']:
            inner_scores = np.full(len(train_idx), np.nan)
            fits = []
            for inner in sorted(set(local_folds.tolist())):
                subtrain = np.where(local_folds != inner)[0]
                valid = np.where(local_folds == inner)[0]
                if set(folds[train_idx[subtrain]]) & {outer, inner}:
                    raise ValueError('Nested fold leakage')
                model = fit_nn_head(train_x[subtrain], y[subtrain, head], alpha, spec)
                inner_scores[valid] = predict_nn_head(model, train_x[valid])
                fits.append({'validation_fold': int(inner), 'training_folds': sorted(set(local_folds[subtrain].tolist())),
                             'training_rows': len(subtrain), 'training_ids_sha256': id_digest(rows, train_idx[subtrain]),
                             'validation_ids_sha256': id_digest(rows, train_idx[valid]),
                             'known_labels': model['known_training_rows'], 'positives': model['positives'],
                             'kind': model['kind'], 'seed_fits': model.get('seed_fits', [])})
            if not np.isfinite(inner_scores).all():
                raise ValueError('Incomplete inner predictions')
            candidates.append({'alpha': float(alpha), 'loss': binary_loss(y[:, head], inner_scores),
                               'scores': inner_scores, 'fits': fits})
        chosen = min(candidates, key=lambda c: (c['loss'], -c['alpha']))  # ties favour stronger regularisation
        scores[:, head] = chosen['scores']
        final = fit_nn_head(train_x, y[:, head], chosen['alpha'], spec)
        final_models.append(final)
        head_history.append({'head': design['head_order'][head], 'chosen_alpha': chosen['alpha'],
                             'candidates': [{k: v for k, v in c.items() if k != 'scores'} for c in candidates]})
    rules = choose_rules(scores, train_rows, design)
    test_scores = np.column_stack([predict_nn_head(m, x[test_idx]) for m in final_models])
    actions = {name: apply_rule(test_scores, selected['rule']) for name, selected in rules.items()}
    weights = {}
    for head, model in zip(design['head_order'], final_models):
        weights.update(head_weights(model, head))
    return {'outer_fold': int(outer), 'method': METHOD, 'training_folds': sorted(set(local_folds.tolist())),
            'training_ids_sha256': id_digest(rows, train_idx), 'test_ids_sha256': id_digest(rows, test_idx),
            'test_indices': test_idx.tolist(), 'models': [strip(m) for m in final_models],
            'head_selection': head_history, 'rules': rules, 'test_scores': test_scores.tolist(),
            'actions': {k: v.tolist() for k, v in actions.items()},
            'inner_validation_scores': scores.tolist(),
            'inner_validation_indices': train_idx.tolist()}, weights


def convergence_summary(fitted_folds):
    """Every network fit that ran, including every inner alpha candidate."""
    total = converged = 0
    iterations, losses, failures = [], [], []
    for fitted in fitted_folds:
        for history in fitted['head_selection']:
            for candidate in history['candidates']:
                for fit in candidate['fits']:
                    for seed_fit in fit.get('seed_fits', []):
                        total += 1
                        converged += int(seed_fit['converged'])
                        iterations.append(seed_fit['iterations'])
                        losses.append(seed_fit['final_training_loss'])
                        if not seed_fit['converged']:
                            failures.append({'outer_fold': fitted['outer_fold'], 'head': history['head'],
                                             'alpha': candidate['alpha'], 'validation_fold': fit['validation_fold'],
                                             'seed': seed_fit['seed'], 'iterations': seed_fit['iterations']})
        for model, head in zip(fitted['models'], [h['head'] for h in fitted['head_selection']]):
            for seed_fit in model.get('seed_fits', []):
                total += 1
                converged += int(seed_fit['converged'])
                iterations.append(seed_fit['iterations'])
                losses.append(seed_fit['final_training_loss'])
                if not seed_fit['converged']:
                    failures.append({'outer_fold': fitted['outer_fold'], 'head': head, 'alpha': model['alpha'],
                                     'validation_fold': None, 'seed': seed_fit['seed'],
                                     'iterations': seed_fit['iterations']})
    return {'network_fits': total, 'converged_before_max_iter': converged,
            'hit_max_iter': total - converged,
            'median_iterations': float(np.median(iterations)) if iterations else None,
            'max_iterations_used': int(max(iterations)) if iterations else None,
            'max_final_training_loss': float(max(losses)) if losses else None,
            'non_converged_fits': failures}
