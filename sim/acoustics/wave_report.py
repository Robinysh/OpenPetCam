"""Self-contained private report for the signed 3-D source-information ladder.

    python3 -m sim.acoustics.wave_report EXPERIMENT --out REPORT.html

Only the explicit HTML artifact is written. Pressure arrays, the solid mask
and animation controls are embedded; viewing needs no file fetch or network.
"""
import argparse
import base64
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
METHODS = {
    'wave_known_source': ('Known source · privileged', 'The exact prescribed signed digital emitter is supplied; one real gain and one common onset are profiled.'),
    'wave_other_recording_template': ('Other recording template', 'A different original recording supplies a signed template. This is an extra source assumption.'),
    'wave_blind32': ('Blind 32 ms source FIR', 'A short causal real source waveform is fitted with the declared ridge prior. Flexibility can absorb room differences.'),
    'wave_power': ('Full wave · power only', 'Full-wave transfer power and spectral ripple; source phase is discarded by this matcher.'),
    'ray_power': ('First-order paths · power', 'Direct and first-order specular paths supply a power-ripple baseline.'),
    'arbitrary_source_control': ('Arbitrary source · control', 'An independent complex source spectrum can explain the recording at many positions. This control must abstain.'),
}


def _load(folder, name):
    return json.loads((folder / name).read_text())


def _num(value, digits=1):
    return '—' if value is None else f'{value:,.{digits}f}'


def _table(headers, rows):
    headers = ''.join(f'<th scope="col">{escape(str(x))}</th>' for x in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(str(x))}</td>' for x in row) + '</tr>' for row in rows)
    return '<div class="table-scroll"><table><thead><tr>' + headers + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def _metrics(rows):
    single = [r for r in rows if r['kind'] == 'single']
    located = [r for r in single if r.get('distance_error_m') is not None]
    null = [r for r in rows if r['kind'] in ('silence', 'background')]
    multiple = [r for r in rows if r['kind'] == 'multiple']
    residuals = [r['normalized_residual'] for r in single
                 if r.get('normalized_residual') is not None and math.isfinite(r['normalized_residual'])]
    return {'n': len(single), 'located': len(located),
            'within1m': sum(r['distance_error_m'] <= 1 for r in located),
            'within30deg': sum(r.get('bearing_error_deg') is not None and r['bearing_error_deg'] <= 30 for r in located),
            'median': float(np.median([r['distance_error_m'] for r in located])) if located else None,
            'p90': float(np.percentile([r['distance_error_m'] for r in located], 90)) if located else None,
            'residual': float(np.median(residuals)) if residuals else None,
            'null_n': len(null), 'null_returned': sum(r.get('position_m') is not None for r in null),
            'multiple_n': len(multiple), 'multiple_returned': sum(r.get('position_m') is not None for r in multiple)}


def _field(path):
    with np.load(path, allow_pickle=False) as arrays:
        times, xs, zs, pressure = (np.asarray(arrays[k], dtype=float) for k in ('times_s', 'x_m', 'z_m', 'pressure'))
        solid = np.asarray(arrays['solid'], dtype=bool)
        source = np.asarray(arrays['source_xz_m'], dtype=float)
        slice_y = float(arrays['slice_y_m'])
    if (times.ndim != 1 or not len(times) or xs.ndim != 1 or len(xs) < 2 or zs.ndim != 1 or len(zs) < 2 or
            pressure.shape != (len(times), len(zs), len(xs)) or solid.shape != pressure.shape[1:] or source.shape != (2,)):
        raise ValueError('3-D field slice arrays have inconsistent dimensions')
    if not all(np.isfinite(a).all() for a in (times, xs, zs, pressure, source)) or not math.isfinite(slice_y):
        raise ValueError('field slice must contain finite values')
    if any((np.diff(a) <= 0).any() for a in (times, xs, zs)):
        raise ValueError('field coordinates/times must increase strictly')
    if not all(np.allclose(np.diff(a), np.diff(a)[0], rtol=1e-8, atol=1e-12) for a in (xs, zs)):
        raise ValueError('field image requires uniform x/z sample coordinates')
    return {'times_s': times.tolist(), 'x_m': xs.tolist(), 'z_m': zs.tolist(),
            'shape': list(pressure.shape), 'pressure_b64': base64.b64encode(pressure.astype('<f8').tobytes()).decode('ascii'),
            'solid_b64': base64.b64encode(solid.astype('u1').tobytes()).decode('ascii'),
            'pressure_encoding': 'float64 little endian, time-z-x order',
            'global_abs_max': float(np.max(np.abs(pressure))), 'source_xz_m': source.tolist(), 'slice_y_m': slice_y}


def render(folder, output=None):
    folder = Path(folder).resolve()
    output = Path(output).resolve() if output else folder.parent / 'index.html'
    protocol = _load(folder, 'protocol.json')
    summary, rooms = _load(folder, 'summary.json'), _load(folder, 'rooms.json')
    raw = _load(folder, 'results.json')
    rows = raw if isinstance(raw, list) else raw['results']
    if summary.get('real_acceptance') is not False or any(r.get('accepted') is not False for r in rows):
        raise ValueError('this report requires explicitly unaccepted offline results')
    cases = _load(folder, 'cases.json')
    field = _field(folder / 'wavefield.npz')
    methods = [name for name in METHODS if name in summary['methods']]
    methods += [name for name in summary['methods'] if name not in methods]
    scenes = list(summary['by_scene'])
    order = [s for s in ('same_grid_control', 'coarse_offgrid_control', 'fine_same_prior', 'fine_2cm', 'fine_10cm_completion') if s in scenes]
    order += [s for s in scenes if s not in order]
    tables = []
    for scene in order:
        table_rows = []
        for method in methods:
            metric = _metrics([r for r in rows if r['scene'] == scene and r['method'] == method])
            table_rows.append([METHODS.get(method, (method, ''))[0], str(metric['n']), str(metric['located']),
                               f'{metric["within1m"]} / {metric["n"]}', f'{metric["within30deg"]} / {metric["n"]}',
                               _num(metric['median']) + ' m', _num(metric['p90']) + ' m', _num(metric['residual'], 3)])
        if scene == 'same_grid_control':
            distinction = 'Same discrete Green dictionary generates and fits these cases: an algorithm check, not independent accuracy.'
        elif scene == 'coarse_offgrid_control':
            distinction = 'Off-grid source positions on the same coarse propagation mesh isolate candidate-grid spacing from numerical-refinement mismatch.'
        else:
            distinction = 'Truth uses a finer numerical grid. Geometry and model mismatch remain explicit simulation assumptions.'
        tables.append(f'<h3>{escape(scene.replace("_", " "))}</h3><p class="muted">{escape(distinction)}</p>' +
                      _table(['Source information', 'Single cases', 'Returned', 'Within 1 m', 'Within 30°',
                              'Median error¹', 'P90 error¹', 'Median residual²'], table_rows))
    null_rows = []
    for method in methods:
        metric = _metrics([r for r in rows if r['method'] == method])
        null_rows.append([METHODS.get(method, (method, ''))[0], f'{metric["null_returned"]} / {metric["null_n"]}',
                          f'{metric["multiple_returned"]} / {metric["multiple_n"]}',
                          str(sum(bool(r.get('accepted')) for r in rows if r['method'] == method))])
    controls = _table(['Method', 'Silence/background coordinates returned', 'Multiple-source coordinates returned',
                       'Real accepted'], null_rows)
    room_rows = []
    for name, metadata in rooms.items():
        room_rows.append([name.replace('_', ' '), str(metadata.get('shape_xyz', '—')),
                          _num(metadata.get('spacing_m'), 3), _num(metadata.get('sound_speed_m_s'), 1),
                          _num(metadata.get('uniform_damping_s_inv'), 1), _num(metadata.get('outer_padding_m'), 2),
                          _num(metadata.get('solid_voxels'), 0)])
    room_table = _table(['Completed model', 'Grid shape', 'Spacing, m', 'Sound speed, m/s', 'Uniform damping, s⁻¹',
                        'Outer padding, m', 'Solid voxels'], room_rows)
    ladder = _table(['Information condition', 'What the method receives'],
                    [METHODS.get(method, (method, 'See frozen protocol.')) for method in methods])
    assumptions = protocol.get('assumptions', [])
    if isinstance(assumptions, dict):
        assumptions = [str(k) + ': ' + str(v) for k, v in assumptions.items()]
    links = []
    for name, title in [('protocol.json', 'Frozen protocol and source hashes'), ('cases.json', 'All declared virtual cases'),
                        ('results.json', 'Every result'), ('summary.json', 'Machine-readable summary'),
                        ('rooms.json', 'Voxel completion assumptions'), ('wavefield.npz', 'Original signed field arrays'),
                        ('source_inventory.json', 'Original recording inventory'), ('engine.json', 'Native engine receipts')]:
        if (folder / name).exists():
            links.append(f'<a href="{escape(os.path.relpath(folder/name, output.parent), quote=True)}">{escape(title)}</a>')
    hashes = protocol.get('input_sha256', {})
    current_match = bool(hashes) and all((ROOT/name).exists() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                                       for name, digest in hashes.items())
    hash_text = ('All frozen input hashes matched at completion and still match at render time.'
                 if summary['input_hashes_match'] and current_match else
                 'Current source hashes do not all match this experiment. Recorded results are preserved; inspect the frozen protocol.')
    references = []
    source_file = folder / 'physics_sources.json'
    if not source_file.exists():
        source_file = folder.parent / 'physics_sources.json'
    if source_file.exists():
        for source in json.loads(source_file.read_text()).get('sources', []):
            url = source['url']
            if not url.startswith('https://'):
                raise ValueError('mathematical reference URL must use https')
            references.append(f'<li><a href="{escape(url,quote=True)}">{escape(source["title"])}</a> — {escape(source.get("use",""))}</li>')
    embedded = json.dumps(field, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    frequencies = protocol.get('frequencies_hz', [80, 220])
    source_count = len(protocol.get('source_waveforms', [])) or summary.get('source_count', 4)
    data = {'RUN': escape(folder.name), 'CASE_COUNT': str(len(cases)), 'SOURCE_COUNT': str(source_count),
            'METHOD_COUNT': str(len(methods)), 'BAND': f'{min(frequencies):g}–{max(frequencies):g} Hz',
            'LADDER': ladder, 'COMPARISONS': ''.join(tables), 'CONTROLS': controls, 'ROOMS': room_table,
            'ASSUMPTIONS': ''.join(f'<li>{escape(str(a))}</li>' for a in assumptions),
            'HASH_TEXT': escape(hash_text), 'LINKS': ''.join(links), 'REFERENCES': ''.join(references),
            'EMBEDDED': embedded, 'SLICE_Y': _num(field['slice_y_m'], 3),
            'REAL_ACCEPTANCE': 'false' if summary.get('real_acceptance') is False else escape(str(summary.get('real_acceptance')))}
    html = TEMPLATE
    for key, value in data.items():
        html = html.replace('{{' + key + '}}', value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    return output


TEMPLATE = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>OpenPetCam · Signed 3-D wave experiment</title><style>
:root{--paper:#f5f3ec;--ink:#293a32;--muted:#607166;--line:#d2dbd0;--green:#386f5a;--blue:#315a97;--red:#b34b33}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.65 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1200px;padding:58px 38px 85px;margin:auto}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400;letter-spacing:-.025em;line-height:1.16}h1{font-size:clamp(42px,6vw,72px);max-width:850px;margin:22px 0}h2{font-size:33px;margin:0 0 20px}h3{font-size:23px;margin:28px 0 14px}p{max-width:80ch;margin:12px 0}.kicker{text-transform:uppercase;letter-spacing:.16em;font-weight:700;font-size:12px;color:var(--muted)}.lede{font-size:21px;max-width:800px}.muted{font-size:14px;color:var(--muted)}.scope{border:1px solid var(--line);display:inline-block;padding:4px 10px;font-size:12px;text-transform:uppercase;letter-spacing:.06em}.notice{border-left:4px solid #af7043;background:#f0e7d7;margin:30px 0;padding:18px 23px}.notice strong{display:block;font-size:20px;margin-bottom:5px}section{border-top:1px solid var(--line);margin-top:42px;padding-top:42px}.note{border-left:2px solid var(--line);padding-left:18px;font-size:14px;color:var(--muted)}a{color:var(--green);text-underline-offset:3px}.table-scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;min-width:710px;width:100%;font-size:14px;font-variant-numeric:tabular-nums}th{color:var(--muted);font-size:12px;text-align:left;font-weight:600;border-bottom:2px solid var(--line);padding:11px 13px}td{padding:12px 13px;border-bottom:1px solid var(--line)}td:first-child,th:first-child{padding-left:0}.field{border:1px solid var(--line);background:#fcfcf7;margin:25px 0}.field-head{padding:16px 20px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:center;gap:15px;font-size:14px}.colorbar{width:180px;height:12px;background:linear-gradient(to right,#315a97,#faf9f2,#b34b33)}.colorlabels{display:flex;justify-content:space-between;font-size:11px;font-variant-numeric:tabular-nums}canvas{width:100%;height:470px;display:block}.controls{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:18px;padding:16px 20px;border-top:1px solid var(--line)}button{border:1px solid #a5b4a7;border-radius:4px;padding:8px 13px;background:#fcfcf7;color:var(--ink);font:14px system-ui;cursor:pointer;min-width:135px}button:hover{background:#e6eee1}button:focus-visible,input:focus-visible{outline:3px solid #d8af63;outline-offset:3px}input[type=range]{width:100%;accent-color:var(--green)}output{font-size:13px;min-width:120px;font-variant-numeric:tabular-nums}.field-note{padding:0 20px 18px;font-size:13px;color:var(--muted)}.assumptions{columns:2;column-gap:40px;padding-left:20px}.assumptions li{font-size:14px;break-inside:avoid;margin-bottom:12px}.evidence{display:flex;gap:10px 24px;flex-wrap:wrap;font-size:14px;margin:24px 0}.references{font-size:14px;padding-left:20px}.references li{margin-bottom:12px}code{font:12px ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}.footer{font-size:13px;color:var(--muted);margin-top:30px}@media(max-width:720px){main{padding:34px 20px 60px}.lede{font-size:19px}section{margin-top:30px;padding-top:32px}canvas{height:380px}.field-head{flex-direction:column;align-items:flex-start}.controls{grid-template-columns:auto 1fr;gap:12px}output{grid-column:1/-1}.assumptions{columns:1}h2{font-size:29px}}
</style></head><body><main><header><div class="kicker">OpenPetCam / Full-wave source experiment</div><h1>How much of the sound must we know?</h1><p class="lede">A wave equation predicts signed pressure. Locating its source is another question. This experiment separates the propagation model from the information supplied about the sound.</p><p class="muted">{{RUN}} · {{BAND}} · {{CASE_COUNT}} virtual cases · {{SOURCE_COUNT}} recorded wavelets · {{METHOD_COUNT}} information conditions</p><span class="scope">Offline numerical experiment · no real coordinates accepted</span></header>
<aside class="notice"><strong>A full 3-D forward calculation, in an assumed room.</strong>The model preserves waveform sign and complex phase. Its enclosing box, wall heights and loss are hypotheses. This is source-position inversion under those hypotheses; geometry gradients and an adjoint are not implemented, so it is not a completed geometry FWI result.</aside>
<section><h2>The pressure field has two signs</h2><p>This is an x–z slice at y = {{SLICE_Y}} m through a genuinely <b>three-dimensional</b> wave calculation. A reciprocal impulse at the assumed microphone location is convolved with a causal 120 Hz Gaussian-windowed packet. The animation shows the stored signed field, not an envelope or a two-dimensional propagation approximation.</p><div class="field"><div class="field-head"><span>3-D pressure field · horizontal slice</span><div><div class="colorbar"></div><div class="colorlabels"><span id="negative"></span><span>0</span><span id="positive"></span></div></div></div><canvas id="field" role="img" aria-label="Animated signed pressure on an x–z slice of a three-dimensional wave simulation"></canvas><div class="controls"><button id="play" type="button">Play slow motion</button><input id="frame" type="range" min="0" value="0" step="1" aria-label="Stored pressure frame"><output id="time"></output></div><div class="field-note">Blue and red represent opposite pressure signs. One global colour scale is shared by every frame; units are arbitrary scalar-wave units, not measured pascals. Dark cells are the model’s solid mask. The cross marks the reciprocal source projected onto this slice. Playback advances one stored frame every 80 ms.</div></div></section>
<section><h2>The source information changes the problem</h2>{{LADDER}}<p>The four source windows are original room recordings used as prescribed virtual emitters. They already contain reverberation and recording-chain effects. Convolving them with another room response is a replay proxy; it does not turn them into dry barks or certified household sounds.</p><p class="note">The known-source condition receives the exact digital waveform used to generate its virtual observation. That is privileged information. Original recordings retain unknown physical source coordinates and unknown historical microphone poses. A simulated <code>true_position_m</code> is a virtual label, not a recovered location in the home.</p></section>
<section><h2>Keep the easy check separate</h2><p>The same-grid condition checks the matching algorithm. Finer-grid truth and altered room models ask a harder question: how much does the result survive numerical and geometric mismatch? Every single-source case remains in the success denominator, including abstentions.</p>{{COMPARISONS}}<p class="muted">¹ Error quantiles describe only returned virtual single-source coordinates; missing predictions have no invented error. Success counts include every single-source case. ² Residuals depend on the method’s source freedom and objective. A lower residual from a flexible source fit does not establish a more accurate location. These repeated source/anchor/scene rows are correlated, so no independent-case confidence interval is claimed.</p></section>
<section><h2>A convincing fit can still return a meaningless point</h2>{{CONTROLS}}<p>Silence and absolute background have no emitting-source coordinate. Multiple-source controls have no single correct location. The table counts diagnostic coordinates returned, even when operational acceptance is false. A candidate is not evidence that an emitting source was detected.</p><p class="note">With arbitrary complex source freedom, a candidate transfer can absorb the observation into its source spectrum. Near-zero residuals at many positions are an expected identifiability failure. The arbitrary-source control always abstains. Real acceptance in this experiment is <b>{{REAL_ACCEPTANCE}}</b>.</p></section>
<section><h2>The room is part of the hypothesis</h2>{{ROOMS}}<p>The disconnected saved segments do not enclose a surveyed room. The wave model adds a rigid rectangular boundary and extrudes the segments to full height, including possible furniture faces. Floor scale has legacy acoustic provenance; ceiling height and volumetric damping remain assumptions. Refinement also changes voxel staircasing and realised outer faces.</p><ul class="assumptions">{{ASSUMPTIONS}}</ul></section>
<section><h2>Inspect the calculation</h2><p class="muted">{{HASH_TEXT}} Pressure frames and controls are embedded, so the report works from a local file without fetching assets.</p><div class="evidence">{{LINKS}}</div><ul class="references">{{REFERENCES}}</ul><p class="muted">These mathematical references support the wave equation, numerical analysis and reciprocity. They do not validate the chosen room, source assumptions or recordings.</p><p class="footer">Private experiment. No camera movement, playback, recording, production-model update or real localization acceptance is implied.</p></section></main>
<script type="application/json" id="data">{{EMBEDDED}}</script><script>
const D=JSON.parse(document.getElementById('data').textContent),canvas=document.getElementById('field'),ctx=canvas.getContext('2d'),slider=document.getElementById('frame'),play=document.getElementById('play'),off=document.createElement('canvas'),oc=off.getContext('2d');
const bytes=Uint8Array.from(atob(D.pressure_b64),c=>c.charCodeAt(0));let pressure;if(new Uint8Array(new Uint16Array([1]).buffer)[0]===1){pressure=new Float64Array(bytes.buffer)}else{pressure=new Float64Array(bytes.length/8);const view=new DataView(bytes.buffer);for(let i=0;i<pressure.length;i++)pressure[i]=view.getFloat64(i*8,true)}
const solid=Uint8Array.from(atob(D.solid_b64),c=>c.charCodeAt(0)),nt=D.shape[0],nz=D.shape[1],nx=D.shape[2],scale=D.global_abs_max||1;off.width=nx;off.height=nz;slider.max=nt-1;let running=false,last=0;
document.getElementById('negative').textContent='−'+D.global_abs_max.toExponential(2);document.getElementById('positive').textContent='+'+D.global_abs_max.toExponential(2);
function draw(){const w=canvas.clientWidth,h=canvas.clientHeight,dpr=Math.min(2,window.devicePixelRatio||1);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr)}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const frame=Number(slider.value),image=oc.createImageData(nx,nz);for(let z=0;z<nz;z++){for(let x=0;x<nx;x++){const cell=z*nx+x,pixel=((nz-1-z)*nx+x)*4;let rgb;if(solid[cell])rgb=[40,61,54];else{const p=Math.max(-1,Math.min(1,pressure[frame*nx*nz+cell]/scale)),a=Math.abs(p),target=p<0?[49,90,151]:[179,75,51];rgb=target.map((v,i)=>Math.round([250,249,242][i]*(1-a)+v*a))}image.data[pixel]=rgb[0];image.data[pixel+1]=rgb[1];image.data[pixel+2]=rgb[2];image.data[pixel+3]=255}}oc.putImageData(image,0,0);const m={left:58,right:25,top:22,bottom:46},pw=w-m.left-m.right,ph=h-m.top-m.bottom,dx=(D.x_m[nx-1]-D.x_m[0])/(nx-1),dz=(D.z_m[nz-1]-D.z_m[0])/(nz-1),x0=D.x_m[0]-dx/2,x1=D.x_m[nx-1]+dx/2,z0=D.z_m[0]-dz/2,z1=D.z_m[nz-1]+dz/2,X=x=>m.left+(x-x0)/(x1-x0)*pw,Y=z=>m.top+(z1-z)/(z1-z0)*ph;ctx.imageSmoothingEnabled=false;ctx.drawImage(off,m.left,m.top,pw,ph);ctx.strokeStyle='#9aac9d';ctx.strokeRect(m.left,m.top,pw,ph);ctx.font='12px system-ui';ctx.fillStyle='#586e60';ctx.textAlign='center';for(let x=Math.ceil(x0);x<=x1;x+=2){ctx.fillText(x+'',X(x),h-m.bottom+21)}ctx.textAlign='right';for(let z=Math.ceil(z0);z<=z1;z+=2){ctx.fillText(z+'',m.left-9,Y(z)+4)}ctx.textAlign='center';ctx.fillText('x, metres',w/2,h-8);ctx.save();ctx.translate(15,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('z, metres',0,0);ctx.restore();const sx=X(D.source_xz_m[0]),sy=Y(D.source_xz_m[1]);if(sx>=m.left&&sx<=w-m.right&&sy>=m.top&&sy<=h-m.bottom){ctx.beginPath();ctx.moveTo(sx-6,sy);ctx.lineTo(sx+6,sy);ctx.moveTo(sx,sy-6);ctx.lineTo(sx,sy+6);ctx.strokeStyle='#ffffff';ctx.lineWidth=4;ctx.stroke();ctx.strokeStyle='#1c3227';ctx.lineWidth=2;ctx.stroke();ctx.lineWidth=1}document.getElementById('time').textContent=(D.times_s[frame]*1000).toFixed(1)+' ms · '+(frame+1)+' / '+nt}
function stop(){running=false;play.textContent='Play slow motion'}function animate(now){if(!running)return;if(!last)last=now;if(now-last>=80){const step=Math.floor((now-last)/80);slider.value=Math.min(nt-1,Number(slider.value)+step);last+=step*80;draw()}if(Number(slider.value)>=nt-1){stop();return}requestAnimationFrame(animate)}
play.addEventListener('click',()=>{if(running){stop();return}if(Number(slider.value)>=nt-1)slider.value=0;running=true;last=0;play.textContent='Pause';draw();requestAnimationFrame(animate)});slider.addEventListener('input',()=>{stop();draw()});window.addEventListener('resize',draw);document.addEventListener('visibilitychange',()=>{if(document.hidden)stop()});draw();
</script></body></html>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    print(render(args.folder, args.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
