import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from objectives import OBJECTIVES, Objective
from optimizers import Run, Settings, export_runs


class TextbookTests(unittest.TestCase):
    def test_derivatives(self):
        eps=1e-5
        for obj in OBJECTIVES.values():
            for x in [np.array([.31,-.48]),np.array([1.15,.75])]:
                eye=np.eye(2)*eps
                numeric=np.array([(obj.value(x+e)-obj.value(x-e))/(2*eps) for e in eye])
                np.testing.assert_allclose(obj.gradient(x),numeric,rtol=2e-8,atol=1e-8)
                h=np.column_stack([(obj.gradient(x+e)-obj.gradient(x-e))/(2*eps) for e in eye])
                np.testing.assert_allclose(obj.hessian(x),h,rtol=1e-8,atol=1e-8)

    def test_fixed_textbook_steps(self):
        run=Run(OBJECTIVES['elongated'],'sd',[.5,.5],Settings('fixed',.01))
        run.step()
        np.testing.assert_allclose(run.x,[.49,.25],atol=1e-14)
        run.step()
        np.testing.assert_allclose(run.x,[.4802,.125],atol=1e-14)

    def test_line_search_and_conjugacy(self):
        obj=OBJECTIVES['rotated']
        sd=Run(obj,'sd',[.8,-.25])
        cg=Run(obj,'cg',[.8,-.25])
        sd.step();cg.step()
        np.testing.assert_allclose(sd.x,[.2429611650485437,-.3737864077669903],atol=1e-14)
        np.testing.assert_allclose(sd.x,cg.x,atol=1e-14)
        self.assertAlmostEqual(sd.records[0]['alpha'],.412621359223301,places=14)
        sd.step();cg.step()
        p0,p1=[np.array(r['direction']) for r in sd.records]
        self.assertAlmostEqual(float(p0@p1),0.,places=13)
        p0,p1=[np.array(r['direction']) for r in cg.records]
        self.assertAlmostEqual(float(p0@obj.matrix@p1),0.,places=13)
        self.assertEqual(len(cg.records),2)
        self.assertTrue(cg.done)
        np.testing.assert_allclose(cg.x,[0,0],atol=1e-14)
        sd.finish()
        self.assertLess(sd.summary()['gradient_norm'],1e-6)
        self.assertGreater(len(sd.records),5)

    def test_newton_one_step(self):
        for key in ['elongated','rotated']:
            obj=OBJECTIVES[key]
            run=Run(obj,'newton',obj.default_start).finish()
            self.assertEqual(len(run.records),1)
            np.testing.assert_allclose(run.x,[0,0],atol=1e-15)

    def test_newton_nonconvex_starts(self):
        obj=OBJECTIVES['nonconvex']
        cases=[([1.5,0],[.55357994,-.55357994],'极小点'),
               ([-1.5,0],[-.41878,.41878],'极小点'),
               ([.75,.75],[-.134797,.134797],'鞍点'),
               ([1.15,.75],[-.41878,.41878],'极小点')]
        for start,expected,state in cases:
            run=Run(obj,'newton',start).finish()
            np.testing.assert_allclose(run.x,expected,atol=4e-6)
            self.assertIn(state,run.status)
            self.assertLess(run.summary()['gradient_norm'],1e-6)
            if start==[1.5,0]:
                np.testing.assert_allclose(run.records[0]['x_new'],[.6086956521739131,-.608695652173913],atol=1e-14)

    def test_learning_rate_stability(self):
        obj=OBJECTIVES['elongated']
        for rate in [.01,.035,.039]:
            run=Run(obj,'sd',[.5,.5],Settings('fixed',rate)).finish()
            self.assertIn('极小点',run.status)
        boundary=Run(obj,'sd',[.5,.5],Settings('fixed',.04,max_steps=200)).finish()
        self.assertIn('最大步数',boundary.status)
        self.assertAlmostEqual(abs(boundary.x[1]),.5,places=12)
        divergent=Run(obj,'sd',[.5,.5],Settings('fixed',.041)).finish()
        self.assertIn('越界',divergent.status)
        self.assertGreater(divergent.history[2]['f'],divergent.history[0]['f'])

    def test_stop_and_validation(self):
        obj=OBJECTIVES['rotated']
        for method in ['sd','newton','cg']:
            run=Run(obj,method,[0,0])
            self.assertTrue(run.done)
            run.step()
            self.assertEqual(len(run.records),0)
        run=Run(obj,'sd',[.8,-.25],Settings(max_steps=1)).finish()
        self.assertIn('最大步数',run.status)
        stalled=Run(obj,'sd',[.8,-.25],Settings('fixed',1e-30)).finish()
        self.assertIn('停滞',stalled.status)
        with self.assertRaises(ValueError): Run(obj,'sd',[np.nan,0])
        with self.assertRaises(ValueError): Run(obj,'sd',[0,0],Settings(tolerance=0))
        with self.assertRaises(ValueError): Run(OBJECTIVES['nonconvex'],'cg',[1,0])
        with self.assertRaises(ValueError): Run(OBJECTIVES['nonconvex'],'sd',[1,0])

    def test_singular_hessian(self):
        class Singular(Objective):
            def value(self,x): return float(x[0])
            def gradient(self,x): return np.array([1.,0.])
            def hessian(self,x): return np.zeros((2,2))
        obj=Singular('singular','test','x1',(0,0),(-1,1,-1,1))
        run=Run(obj,'newton',[0,0]).finish()
        self.assertIn('奇异',run.status)
        self.assertEqual(len(run.records),0)
        textbook=Run(OBJECTIVES['nonconvex'],'newton',[0,1/np.sqrt(3)]).finish()
        self.assertIn('奇异',textbook.status)

    def test_backtracking_extension(self):
        run=Run(OBJECTIVES['nonconvex'],'sd',[1.5,0],Settings('backtrack')).finish()
        values=[s['f'] for s in run.history]
        self.assertTrue(all(b<=a+1e-12 for a,b in zip(values,values[1:])))
        self.assertIn('极小点',run.status)

    def test_export_and_accounting(self):
        run=Run(OBJECTIVES['rotated'],'cg',[.8,-.25]).finish()
        self.assertEqual(run.counts,{'value':3,'gradient':3,'hessian':3})
        with tempfile.TemporaryDirectory() as folder:
            export_runs(folder,[run])
            with Path(folder,'iterations.csv').open(encoding='utf-8-sig') as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),3)
            self.assertEqual(rows[-1]['alpha_to_next'],'')
            data=json.loads(Path(folder,'experiment.json').read_text(encoding='utf-8'))
            self.assertEqual(data['runs'][0]['summary']['steps'],2)
            self.assertEqual(len(data['runs'][0]['transitions']),2)


if __name__=='__main__':
    unittest.main(verbosity=2)
