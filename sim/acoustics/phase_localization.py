"""Offline phase/source-knowledge information ladder, not real localization proof.

Known-source fitting profiles one real gain and a bounded common onset. Blind
fitting instead declares a short causal real source FIR/basis and ridge penalty.
An arbitrary complex source per frequency is an identifiability CONTROL, never
a localizer. All outputs retain every candidate, abstain on spatial ties, and
keep operational acceptance false. No data, hardware, model or file access.
"""
import math

import numpy as np


MAX_ONSETS = 129
MAX_SOURCE_LENGTH = 128


def _real_vector(value, name):
    source = np.asarray(value)
    if source.dtype.kind not in 'biuf' or source.ndim != 1 or not len(source):
        raise ValueError(name + ' must be a nonempty real vector')
    result = source.astype(float)
    if not np.isfinite(result).all():
        raise ValueError(name + ' must be finite')
    return result


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be positive and finite')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + ' must be positive and finite') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be positive and finite')
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def _onsets(value):
    values = _real_vector(value, 'onsets_s')
    if len(values) > MAX_ONSETS or np.max(np.abs(values)) > 10:
        raise ValueError('onset search must contain at most129 values within +/-10seconds')
    return np.unique(values)


def _inputs(observed, green, frequencies_hz, band_hz=None, frequency_weights=None):
    y, g = np.asarray(observed, complex), np.asarray(green, complex)
    f = _real_vector(frequencies_hz, 'frequencies_hz')
    if y.shape != f.shape or g.ndim != 2 or g.shape[1] != len(f) or not 1 <= len(g) <= 8192:
        raise ValueError('observed/source frequency vectors and candidate Green matrix must share a grid')
    if not np.isfinite(y).all() or not np.isfinite(g).all() or np.any(f < 0) or np.any(np.diff(f) <= 0):
        raise ValueError('finite spectra and strictly increasing nonnegative frequencies required')
    weights = np.ones(len(f)) if frequency_weights is None else _real_vector(frequency_weights, 'frequency_weights')
    if weights.shape != f.shape or np.any(weights < 0):
        raise ValueError('frequency weights must match the grid and be nonnegative')
    keep = weights > 0
    if band_hz is not None:
        band = _real_vector(band_hz, 'band_hz')
        if len(band) != 2 or not 0 <= band[0] < band[1]:
            raise ValueError('band_hz must be an increasing nonnegative pair')
        keep &= (f >= band[0]) & (f <= band[1])
    if not keep.any():
        raise ValueError('comparison band has no positively weighted frequencies')
    y, g, f, weights = y[keep], g[:, keep], f[keep], weights[keep]
    # Relative scaling is essential: physical Green functions can be tiny.
    scale = float(np.max(np.abs(y)))
    scaled = y / scale if scale else y.copy()
    energy = float(np.sum(weights * abs(scaled)**2))
    return scaled, g, f, weights, scale, energy, keep


def _finish(method, profiles, residuals, objectives, onsets, frequency, observation_scale,
            energy, tie_tolerance, metadata):
    tolerance = float(tie_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tie_tolerance must be finite and nonnegative')
    valid = [i for i, row in enumerate(profiles) if row['objective'] is not None]
    minimum = min((profiles[i]['objective'] for i in valid), default=None)
    tied = [i for i in valid if profiles[i]['objective'] <= minimum + tolerance] if valid else []
    informative = bool(energy and valid and min(profiles[i]['normalized_residual'] for i in valid) < 1 - 1e-12)
    best = tied[0] if informative and len(tied) == 1 else None
    status = ('no_observed_energy' if not energy else 'uninformative' if not informative else
              'ambiguous' if len(tied) != 1 else 'conditional_candidate')
    return {'method': method, 'status': status, 'accepted': False,
            'real_world_validated': False, 'best_index': best,
            'best_onset_s': profiles[best]['onset_s'] if best is not None else None,
            'normalized_residual': profiles[best]['normalized_residual'] if best is not None else None,
            'minimum_objective': minimum, 'near_best_indices': tied,
            'tie_tolerance': tolerance, 'profiles': profiles,
            'normalized_residual_profiles': residuals, 'objective_profiles': objectives,
            'onsets_s': onsets.tolist(), 'frequencies_hz': frequency.tolist(),
            'comparison_frequency_bins': len(frequency), 'observation_spectrum_scale': observation_scale,
            'energy_policy': 'relative spectral scaling; only exact zero observation energy is a silence null',
            'frequency_grid_contract': 'Shared physical frequencies and consistent DFT windows; caller verifies continuity, convolution padding and onset bounds.',
            'uncertainty_kind': 'descriptive objective ties/sensitivity, not calibrated confidence', **metadata}


def _blank(index, active):
    return {'candidate_index': index, 'active_frequency_bins': int(active),
            'objective': None, 'normalized_residual': None, 'onset_s': None,
            'gain': None, 'status': 'insufficient_active_support'}


def known_source_spectra(observed_spectrum, source_spectrum, green, frequencies_hz, *,
                         onsets_s=(0.,), positive_gain=True, band_hz=None,
                         frequency_weights=None, min_active_bins=8, tie_tolerance=1e-8):
    """Profile one REAL gain and one common time shift across all frequencies.

    Spectra on identical physical frequencies permit comparison of native
    simulations with different sampling rates. No extra dt multiplier is used.
    Source knowledge is privileged information, not available for arbitrary barks.
    """
    if type(positive_gain) is not bool:
        raise ValueError('positive_gain must declare a boolean source-polarity policy')
    minimum = _integer(min_active_bins, 'min_active_bins', 1, 1000000)
    source = np.asarray(source_spectrum, complex)
    if source.shape != np.shape(observed_spectrum) or not np.isfinite(source).all():
        raise ValueError('finite source spectrum on the observed frequency grid required')
    y, greens, f, weights, yscale, energy, keep = _inputs(observed_spectrum, green, frequencies_hz,
                                                        band_hz, frequency_weights)
    source = source[keep]
    sscale = float(np.max(np.abs(source)))
    source = source / sscale if sscale else source
    onsets = _onsets(onsets_s)
    phases = np.exp(-2j * np.pi * onsets[:, None] * f)
    profiles, residuals, objectives = [], [], []
    for index, g in enumerate(greens):
        gscale = float(np.max(np.abs(g)))
        template = source * (g / gscale) if gscale else np.zeros_like(g)
        active = np.count_nonzero(np.abs(template) > np.max(np.abs(template)) * 1e-12) if np.any(template) else 0
        row = _blank(index, active)
        rr, oo = [None] * len(onsets), [None] * len(onsets)
        if energy and sscale and gscale and active >= minimum:
            p = phases * template
            denominator = float(np.sum(weights * abs(template)**2))
            numerator = np.real(np.sum(weights * np.conj(p) * y, axis=1))
            gains = numerator / denominator
            if positive_gain:
                gains = np.maximum(gains, 0)
            values = np.maximum(0, (energy - 2 * gains * numerator + gains**2 * denominator) / energy)
            chosen = int(np.argmin(values))
            rr = values.tolist(); oo = values.tolist()
            physical_gain = float((gains[chosen] * yscale / sscale) / gscale)
            row.update(objective=float(values[chosen]), normalized_residual=float(values[chosen]),
                       onset_s=float(onsets[chosen]), gain=physical_gain if math.isfinite(physical_gain) else None,
                       status='profiled', tied_onsets_s=onsets[values <= values[chosen] + tie_tolerance].tolist())
        profiles.append(row); residuals.append(rr); objectives.append(oo)
    return _finish('known_source_real_gain_common_onset', profiles, residuals, objectives, onsets, f,
                   yscale, energy, tie_tolerance,
                   {'source_knowledge': 'declared digital waveform; privileged known-source comparison',
                    'gain_policy': 'nonnegative real' if positive_gain else 'signed real; source polarity not constrained',
                    'frequency_phase_policy': 'one linear onset phase ramp; no fitted per-frequency phase'})


def smooth_source_basis(length, columns):
    """Orthonormal low-order cosine basis on a finite causal source FIR."""
    _integer(length, 'length', 1, MAX_SOURCE_LENGTH)
    _integer(columns, 'columns', 1, length)
    basis = np.cos(np.pi * (np.arange(length)[:, None] + .5) * np.arange(columns) / length)
    basis[:, 0] *= math.sqrt(1 / length)
    if columns > 1:
        basis[:, 1:] *= math.sqrt(2 / length)
    return basis


def blind_fir_spectra(observed_spectrum, green, frequencies_hz, source_sample_rate, *,
                      source_length=16, source_basis=None, ridge=.001, onsets_s=(0.,),
                      band_hz=None, frequency_weights=None, min_active_bins=8, tie_tolerance=1e-8):
    """Variable projection of a causal REAL source wavelet, with explicit prior.

    Per-candidate Green normalization removes unknown source-level/range gain.
    Ridge is dimensionless relative to design-column energy. Flexible FIRs can
    absorb room differences; this is a conditional source prior, not an oracle.
    """
    length = _integer(source_length, 'source_length', 1, MAX_SOURCE_LENGTH)
    rate = _positive(source_sample_rate, 'source_sample_rate')
    penalty = float(ridge)
    if not math.isfinite(penalty) or not 0 <= penalty <= 1e6:
        raise ValueError('ridge must be finite in0..1e6')
    minimum = _integer(min_active_bins, 'min_active_bins', 1, 1000000)
    if source_basis is not None and np.asarray(source_basis).dtype.kind not in 'biuf':
        raise ValueError('source_basis must be real numeric coefficients')
    basis = np.eye(length) if source_basis is None else np.asarray(source_basis, float)
    if basis.ndim != 2 or basis.shape[0] != length or not 1 <= basis.shape[1] <= length or \
            not np.isfinite(basis).all() or np.linalg.matrix_rank(basis) != basis.shape[1]:
        raise ValueError('source_basis must be a finite full-rank real causal length-by-coefficient matrix')
    y, greens, f, weights, yscale, energy, _ = _inputs(observed_spectrum, green, frequencies_hz,
                                                     band_hz, frequency_weights)
    if np.max(f) > rate / 2 + 1e-9:
        raise ValueError('frequencies exceed the declared source FIR sample rate')
    onsets = _onsets(onsets_s)
    columns = basis.shape[1]
    if len(greens) * len(onsets) * columns**3 > 1000000000:
        raise ValueError('declared source-projection work exceeds the bounded offline budget')
    fourier = np.exp(-2j * np.pi * f[:, None] * np.arange(length) / rate) @ basis
    profiles, residuals, objectives = [], [], []
    for index, g in enumerate(greens):
        gscale = float(np.max(np.abs(g)))
        gn = g / gscale if gscale else np.zeros_like(g)
        active = np.count_nonzero(np.abs(gn) > 1e-12)
        row = _blank(index, active)
        rr, oo, coefficients, ranks = [None] * len(onsets), [None] * len(onsets), [], []
        if energy and gscale and active >= minimum:
            rr, oo = [], []
            base = gn[:, None] * fourier
            gram = np.real(base.conj().T @ (weights[:, None] * base))
            column_energy = float(np.trace(gram) / columns)
            regularization = penalty * column_energy
            weighted_base = np.concatenate((base.real, base.imag)) * np.sqrt(np.r_[weights, weights])[:, None]
            singular = np.linalg.svd(weighted_base, compute_uv=False)
            data_rank = int(np.sum(singular > singular[0] * 1e-10)) if singular[0] else 0
            for onset in onsets:
                a = base * np.exp(-2j * np.pi * f[:, None] * onset)
                rhs = np.real(a.conj().T @ (weights * y))
                if regularization:
                    coef = np.linalg.solve(gram + regularization * np.eye(columns), rhs)
                    rank = data_rank
                else:
                    stacked = np.concatenate((a.real, a.imag)) * np.sqrt(np.r_[weights, weights])[:, None]
                    target = np.r_[y.real, y.imag] * np.sqrt(np.r_[weights, weights])
                    coef, _, rank, _ = np.linalg.lstsq(stacked, target, rcond=1e-10)
                residual = float(np.sum(weights * abs(y - a @ coef)**2) / energy)
                objective = residual + regularization * float(coef @ coef) / energy
                coefficients.append(coef); ranks.append(int(rank)); rr.append(residual); oo.append(objective)
            chosen = int(np.argmin(oo))
            scale = yscale / gscale
            waveform = (basis @ coefficients[chosen]) * scale if math.isfinite(scale) else None
            row.update(objective=oo[chosen], normalized_residual=rr[chosen], onset_s=float(onsets[chosen]),
                       source_fir=waveform.tolist() if waveform is not None and np.isfinite(waveform).all() else None,
                       source_design_rank=ranks[chosen], status='profiled',
                       source_design_singular_values=singular.tolist(),
                       tied_onsets_s=onsets[np.asarray(oo) <= oo[chosen] + tie_tolerance].tolist())
        profiles.append(row); residuals.append(rr); objectives.append(oo)
    return _finish('blind_causal_real_fir_variable_projection', profiles, residuals, objectives, onsets, f,
                   yscale, energy, tie_tolerance,
                   {'source_knowledge': 'unknown source constrained to declared causal FIR/basis',
                    'source_length': length, 'source_sample_rate': rate, 'ridge': penalty,
                    'source_basis': basis.tolist(), 'source_basis_columns': columns,
                    'gain_policy': 'source level absorbed into real FIR; Green amplitude normalized per candidate'})


def arbitrary_source_control_spectra(observed_spectrum, green, frequencies_hz, *,
                                     band_hz=None, frequency_weights=None, min_active_bins=8,
                                     tie_tolerance=1e-8):
    """Unconstrained complex S(f) can explain Y wherever G(f) is nonzero.

    No source spectrum is estimated or used as localization truth. A unique
    dictionary minimum is still not a claimed coordinate under this control.
    """
    minimum = _integer(min_active_bins, 'min_active_bins', 1, 1000000)
    y, greens, f, weights, scale, energy, _ = _inputs(observed_spectrum, green, frequencies_hz,
                                                    band_hz, frequency_weights)
    profiles, residuals = [], []
    compatible = []
    for index, g in enumerate(greens):
        active = np.abs(g) > 0
        row = _blank(index, active.sum())
        value = None
        if energy and active.sum() >= minimum:
            value = float(np.sum(weights[~active] * abs(y[~active])**2) / energy)
            row.update(objective=value, normalized_residual=value, onset_s=0., status='arbitrary_source_absorbs_transfer')
            if value <= tie_tolerance:
                compatible.append(index)
        profiles.append(row); residuals.append([value])
    result = _finish('arbitrary_complex_source_identifiability_control', profiles, residuals, residuals,
                     np.array([0.]), f, scale, energy, tie_tolerance,
                     {'compatible_indices': compatible, 'source_knowledge': 'arbitrary independent complex value at every frequency'})
    result.update(best_index=None, best_onset_s=None, normalized_residual=None,
                  status='source_unidentifiable' if energy else 'no_observed_energy',
                  reason='Source freedom absorbs nonzero Green responses; this CONTROL never returns a coordinate.')
    return result


def _waveform_inputs(observed, green, sample_rate, green_kind, nfft, extra_length=0):
    observed, rate = _real_vector(observed, 'observed'), _positive(sample_rate, 'sample_rate')
    values = np.asarray(green)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError('Green dictionary must be a nonempty matrix')
    if green_kind == 'ir':
        if values.dtype.kind not in 'biuf' or not np.isfinite(values).all():
            raise ValueError('IR dictionary must contain finite real samples')
        required = max(len(observed), values.shape[1] + extra_length)
        size = 1 << (required - 1).bit_length() if nfft is None else nfft
        _integer(size, 'nfft', required, 1048576)
        transfers = np.fft.rfft(values, n=size, axis=1)
    elif green_kind == 'transfer':
        size = 2 * (values.shape[1] - 1) if nfft is None else nfft
        _integer(size, 'nfft', len(observed), 1048576)
        if size // 2 + 1 != values.shape[1]:
            raise ValueError('transfer dictionary does not match the declared nfft')
        transfers = values
    else:
        raise ValueError('green_kind must be transfer or ir')
    frequency = np.fft.rfftfreq(size, 1 / rate)
    weights = np.full(len(frequency), 2.)
    weights[0] = 1
    if size % 2 == 0:
        weights[-1] = 1
    return np.fft.rfft(observed, n=size), transfers, frequency, weights, size


def known_source(observed, source, green, sample_rate, *, green_kind='transfer', nfft=None, **options):
    source = _real_vector(source, 'source')
    y, g, f, weights, size = _waveform_inputs(observed, green, sample_rate, green_kind, nfft, len(source) - 1)
    if len(source) > size:
        raise ValueError('source exceeds the declared FFT window')
    return known_source_spectra(y, np.fft.rfft(source, n=size), g, f, frequency_weights=weights, **options)


def blind_fir(observed, green, sample_rate, *, green_kind='transfer', nfft=None, source_length=16, **options):
    _integer(source_length, 'source_length', 1, MAX_SOURCE_LENGTH)
    y, g, f, weights, _ = _waveform_inputs(observed, green, sample_rate, green_kind, nfft, source_length - 1)
    return blind_fir_spectra(y, g, f, sample_rate, source_length=source_length,
                              frequency_weights=weights, **options)


def arbitrary_source_control(observed, green, sample_rate, *, green_kind='transfer', nfft=None, **options):
    y, g, f, weights, _ = _waveform_inputs(observed, green, sample_rate, green_kind, nfft)
    return arbitrary_source_control_spectra(y, g, f, frequency_weights=weights, **options)
