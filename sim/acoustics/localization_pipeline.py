"""Reusable frozen LOW-band source-prior/refinement research pipeline.

Inputs are derived burst powers and a forward-model dictionary. No source truth
enters matching, harmonic attenuation or adaptive candidate placement. Matching
remains conditional; this class never authorizes a physical camera action.
"""
from collections import OrderedDict

import numpy as np

from .localization import rank_candidates, ripple_features
from .localization_adaptive import distinct_centers, local_positions
from .localization_harmonic import attenuate_lines
from .localization_resolution import LOW, dictionary as build_dictionary


class ResearchMatcher:
    def __init__(self, model, positions, features, receiver, frequency):
        self.model = model
        self.positions, self.features = np.asarray(positions), np.asarray(features)
        self.receiver, self.frequency = np.asarray(receiver), np.asarray(frequency)
        endpoints = np.asarray([point for wall in model.declared_wall_extents
            for point in (wall['a_xz_m'], wall['b_xz_m'])])
        if endpoints.ndim != 2 or endpoints.shape[1:] != (2,):
            raise ValueError('declared endpoint bounds required')
        self.bounds = (float(endpoints[:, 0].min()), float(endpoints[:, 0].max()),
                       float(endpoints[:, 1].min()), float(endpoints[:, 1].max()))
        self.cache = OrderedDict()

    def match(self, burst_powers, *, harmonic=False, refine=False, progress=lambda *args: None):
        powers = np.asarray(burst_powers, float)
        if (powers.ndim != 2 or len(powers) < 1 or powers.shape[1:] != self.frequency.shape
                or not np.isfinite(powers).all() or np.any(powers < 0)):
            raise ValueError('finite nonnegative burst spectra matching the frequency grid required')
        progress()
        diagnostics, changed = [], []
        for power in powers:
            if harmonic:
                result, diagnostic, _ = attenuate_lines(power, self.frequency)
                changed.append(result); diagnostics.append(diagnostic)
            else:
                changed.append(power)
        mean_power = np.mean(changed, axis=0)
        observed, rms = ripple_features(mean_power, self.frequency,
            **{key: value for key, value in LOW.items() if key != 'name'})
        global_rank = rank_candidates(observed, self.features, self.positions)
        centers, local = [], np.empty((0, 3))
        final = dict(global_rank)
        if refine:
            centers = distinct_centers(global_rank['scores'], self.positions)
            local = local_positions(self.positions[centers], self.positions, self.bounds)
            missing = np.asarray([point for point in local if tuple(point) not in self.cache], float).reshape((-1, 3))
            if len(missing):
                new, _, _ = build_dictionary(self.model, missing, self.receiver, self.frequency, progress)
                for point, feature in zip(missing, new):
                    self.cache[tuple(point)] = feature
            if len(local):
                local_features = np.asarray([self.cache[tuple(point)] for point in local])
                final = rank_candidates(observed, np.concatenate((self.features, local_features)),
                                        np.concatenate((self.positions, local)))
            while len(self.cache) > 8192:
                self.cache.popitem(last=False)
        for rank in (global_rank, final):
            rank.pop('scores', None)
            rank['near_best_count'] = len(rank.pop('near_best_candidates', []))
        return {'global': global_rank, 'final': final, 'new_candidates_m': local.tolist(),
                'global_center_indexes': centers, 'feature_ripple_rms': float(rms),
                'gated_bursts': sum(detail['gated'] for detail in diagnostics),
                'mean_masked_fraction': float(np.mean([detail['low_band_masked_fraction'] for detail in diagnostics])) if diagnostics else 0.}
