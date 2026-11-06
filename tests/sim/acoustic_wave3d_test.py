"""Independent physics checks for the new signed 3-D wave engine."""
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.wave3d import Grid, simulate, _run_bounded


@unittest.skipUnless(shutil.which('clang++') or shutil.which('c++'), 'local C++ compiler required')
class Wave3DTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def run_wave(self, grid, source, drive, receivers, **kwargs):
        return simulate(grid, source, drive, receivers, cache_dir=self.cache.name, timeout_s=30, **kwargs)

    def test_free_field_3d_delay_inverse_distance_and_absolute_source_scaling(self):
        grid = Grid((0, 0, 0), (45, 45, 45), .025)
        t = np.arange(70) * grid.dt_s
        drive = np.exp(-.5*((t-.0009)/.00018)**2)
        source = np.array([.55, .55, .55])
        distances = np.array([.15, .25])
        receivers = source + np.column_stack((distances, np.zeros(2), np.zeros(2)))
        result = self.run_wave(grid, source, drive, receivers)
        # Continuous 3-D Green's solution for this exact PDE, before walls:
        # p(r,t) = q(t-r/c)/(4*pi*c^2*r). No fitted gains or alignment.
        for k, distance in enumerate(distances):
            expected = np.exp(-.5*((t-distance/343-.0009)/.00018)**2)/(4*np.pi*343**2*distance)
            keep = (t>.001) & (t<.0022)
            error = np.linalg.norm(result.signals[k,keep]-expected[keep])/np.linalg.norm(expected[keep])
            self.assertLess(error, .13)
            self.assertAlmostEqual(t[np.argmax(result.signals[k])], .0009+distance/343, delta=2*grid.dt_s)
        ratio = result.signals[0].max()/result.signals[1].max()
        self.assertAlmostEqual(ratio, distances[1]/distances[0], delta=.15)
        self.assertEqual(result.signals.shape, (2, len(t)))
        self.assertFalse(result.metadata['real_room_validated'])

    def modal_run(self, shape, h):
        grid = Grid((0,0,0), shape, h, courant=.45)
        modes = np.array([1,2,1])
        ijk = np.indices(shape).reshape(3,-1).T
        phi = np.prod(np.cos(np.pi*modes*(ijk+.5)/np.asarray(shape)), axis=1)
        drive = np.zeros(260); drive[0]=1/grid.dt_s
        receiver = (np.array(shape)//3)*h
        result = self.run_wave(grid, ijk*h, drive, receiver, source_weights=phi*h**3)
        probe_index = np.array(shape)//3
        probe_phi = np.prod(np.cos(np.pi*modes*(probe_index+.5)/np.asarray(shape)))
        a = result.signals[0]/probe_phi
        theta = 2*np.arcsin(grid.courant*np.sqrt(np.sum(np.sin(np.pi*modes/(2*np.asarray(shape)))**2)))
        expected = grid.dt_s*np.sin(np.arange(len(a))*theta)/np.sin(theta)
        np.testing.assert_allclose(a, expected, atol=1e-15, rtol=2e-11)
        select = np.flatnonzero(np.abs(a[1:-1])>.2*np.max(np.abs(a)))+1
        cosine = np.median((a[select+1]+a[select-1])/(2*a[select]))
        measured_omega = np.arccos(cosine)/grid.dt_s
        continuous = 343*np.pi*np.linalg.norm(modes/(np.asarray(shape)*h))
        return abs(measured_omega-continuous)/continuous

    def test_exact_3d_lattice_mode_dispersion_and_refinement(self):
        coarse = self.modal_run((12,13,14), .04)
        fine = self.modal_run((24,26,28), .02)
        self.assertGreater(coarse, 1e-4)
        self.assertLess(fine, coarse/3.5)

    def test_offgrid_reciprocity_with_solid_and_nonuniform_damping(self):
        solid = np.zeros((22,21,20), bool); solid[12:14,4:8,4:8]=True
        sigma = np.indices(solid.shape)[0]*2.
        grid=Grid((0,0,0),solid.shape,.03,solid=solid,sigma=sigma)
        drive=np.zeros(160);drive[0]=1/grid.dt_s
        a,b=(.235,.222,.197),(.433,.383,.312)
        ab=self.run_wave(grid,a,drive,b).signals[0]
        ba=self.run_wave(grid,b,drive,a).signals[0]
        self.assertGreater(np.linalg.norm(ab),0)
        np.testing.assert_allclose(ab,ba,atol=1e-16,rtol=2e-12)

    def test_rigid_wall_has_positive_reflection_and_no_through_wall(self):
        shape=(61,41,41);solid=np.zeros(shape,bool);solid[31,:,:]=True
        free=Grid((0,0,0),shape,.02);wall=Grid((0,0,0),shape,.02,solid=solid)
        t=np.arange(110)*free.dt_s
        drive=np.exp(-.5*((t-.0008)/.00012)**2)
        source=(.4,.4,.4);receivers=[(.44,.4,.4),(.8,.4,.4)]
        clear=self.run_wave(free,source,drive,receivers).signals
        blocked=self.run_wave(wall,source,drive,receivers).signals
        np.testing.assert_array_equal(blocked[1],0)
        self.assertGreater(np.linalg.norm(clear[1]),1e-8)
        echo=blocked[0]-clear[0]
        expected=.0008+(2*.61-.4-.44)/343  # wall face is half a cell before solid node31
        keep=np.abs(t-expected)<.00025
        self.assertGreater(echo[keep].max(),1e-7)
        self.assertGreater(echo[keep].max(),abs(echo[keep].min()))
        self.assertAlmostEqual(t[np.argmax(echo*keep)],expected,delta=3*free.dt_s)

    def test_signed_source_linearity_and_no_input_mutation(self):
        grid=Grid((0,0,0),(13,14,15),.04)
        drive=np.zeros(100);drive[2:5]=[1,-2,1];before=drive.copy()
        first=self.run_wave(grid,(.2,.2,.2),drive,(.32,.28,.24)).signals
        second=self.run_wave(grid,(.2,.2,.2),-2.3*drive,(.32,.28,.24)).signals
        np.testing.assert_allclose(second,-2.3*first,rtol=1e-12,atol=1e-18)
        np.testing.assert_array_equal(drive,before)
        self.assertTrue((first<0).any());self.assertTrue((first>0).any())

    def test_lossless_discrete_energy_conserves_and_diagonal_damping_decays(self):
        free=Grid((0,0,0),(13,14,15),.035)
        damped=Grid((0,0,0),free.shape_xyz,.035,sigma=100.)
        drive=np.zeros(500);drive[:2]=[1/free.dt_s,-1/free.dt_s]
        a=self.run_wave(free,(.21,.21,.21),drive,(.28,.28,.28),energy_stride=2)
        b=self.run_wave(damped,(.21,.21,.21),drive,(.28,.28,.28),energy_stride=2)
        self.assertGreater(a.energy[5],0)
        np.testing.assert_allclose(a.energy[5:],a.energy[5],rtol=2e-12)
        self.assertGreater(b.energy[5],0)
        self.assertLess(b.energy[-1],b.energy[5]*.1)
        self.assertLessEqual(np.diff(b.energy[5:]).max(),b.energy[5]*1e-12)

    def test_cfl_bad_shapes_solid_sampling_and_resources_rejected(self):
        for kwargs in [dict(courant=.58),dict(courant=0),dict(spacing_m=0),
                       dict(sigma=-1),dict(shape_xyz=(2,3,4)),dict(shape_xyz=(10000,10000,10000))]:
            values=dict(origin_m=(0,0,0),shape_xyz=(8,8,8),spacing_m=.1);values.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):Grid(**values)
        solid=np.zeros((8,8,8),bool);solid[3,3,3]=True
        grid=Grid((0,0,0),solid.shape,.1,solid=solid)
        for source,probe in [((.31,.3,.3),(.5,.5,.5)),((.1,.1,.1),(.3,.3,.3)),((-1,0,0),(.5,.5,.5))]:
            with self.subTest(source=source,probe=probe),self.assertRaises(ValueError):
                self.run_wave(grid,source,[1,0,0],probe)

    def test_bounded_process_timeout_kills_run(self):
        start=time.monotonic()
        with self.assertRaises(TimeoutError):
            _run_bounded([sys.executable,'-c','import time; time.sleep(2)'],.03)
        self.assertLess(time.monotonic()-start,1.)

    def test_exact_free_node_next_to_solid_survives_float_roundoff(self):
        origin=np.array([-.807,.13,-.22]);h=.035
        shape=tuple(np.array([8,9,10],dtype=np.int64))
        solid=np.zeros(shape,bool);solid[3,3,4]=True
        grid=Grid(tuple(origin),shape,h,solid=solid)
        node=origin+np.array([2,3,4])*h
        result=self.run_wave(grid,node,[1,0,0,0],node)
        self.assertGreater(result.signals[0,1],0)


if __name__=='__main__':
    unittest.main()
