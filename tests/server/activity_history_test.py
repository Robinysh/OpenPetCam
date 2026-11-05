"""Historical activity uses retained detections without rewriting original evidence."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV',str(ROOT/'config/site.env.example'))
sys.path.insert(0,str(ROOT/'server'))
import app

class ActivityHistory(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=str(Path(self.temp.name)/'data.db')
        self.con=sqlite3.connect(self.path)
        self.con.executescript('CREATE TABLE minute(ts INTEGER PRIMARY KEY,n INT,present INT,moving INT,person INT,motion REAL,noise_db REAL,sounds INT,snap TEXT); CREATE TABLE obs(ts REAL,score REAL,moving INT); CREATE TABLE events(id INT,ts REAL,kind TEXT,detail TEXT,clip TEXT,thumb TEXT,label TEXT);')
        self.addCleanup(self.con.close)
        self.patch=patch.object(app,'DB_PATH',self.path);self.patch.start();self.addCleanup(self.patch.stop)
        self.bounds=patch.object(app,'day_bounds',return_value=('2026-01-01',0,86400));self.bounds.start();self.addCleanup(self.bounds.stop)
    def minute(self,t,present=2,moving=0):
        self.con.execute('INSERT INTO minute VALUES (?,2,?,?,0,0,NULL,0,NULL)',(t,present,moving));self.con.commit()
    def test_absent_debounce_is_removed_only_when_complete_observations_survive(self):
        self.minute(60);self.minute(120)
        self.con.executemany('INSERT INTO obs VALUES (?,0,0)',[(61,),(64,),(121,) ]);self.con.commit()
        r=app.day_report()
        self.assertEqual(r['visible_min'],1)
        self.assertEqual(self.con.execute('SELECT sum(present) FROM minute').fetchone()[0],4)
        self.assertEqual(app.day_metrics('2026-01-01')['visible'],1)
    def test_missing_minutes_do_not_extend_a_rest(self):
        for t in (60,120,600):self.minute(t)
        self.assertEqual(app.day_report()['longest_rest_min'],2)
        self.assertEqual(app.day_metrics('2026-01-01')['longest_rest'],2)

if __name__=='__main__':unittest.main()
