"""Fresh-process 30-step gates. No evidence is reused from older source revisions."""
import json,os,pathlib,shutil,subprocess,sys,tempfile,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
TOKENIZER=pathlib.Path(os.environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER for full operational gates')
class OperationalGates(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  (ROOT/'evidence').mkdir(exist_ok=True);cls.tmp=tempfile.TemporaryDirectory(prefix='gates_',dir=ROOT/'evidence');cls.root=pathlib.Path(cls.tmp.name);cls.device='cuda' if torch.cuda.is_available() else 'cpu'
  off=cls.root/'xai_off.yaml';off.write_text((ROOT/'configs/tiny_moe.yaml').read_text()+'\nexplainability:\n  enabled: false # source: G5 zero-perturbation verification\n')
  def run(name,extra=()):
   command=[sys.executable,str(ROOT/'harness/train.py'),'--tokenizer',str(TOKENIZER),'--seed','1234','--device',cls.device,'--out',str(cls.root/name),*map(str,extra)]
   result=subprocess.run(command,capture_output=True,text=True,encoding='utf8');(cls.root/(name+'_console.txt')).write_text(result.stdout+result.stderr)
   if result.returncode:raise RuntimeError(name+': '+result.stdout+result.stderr)
  run('full_a');run('full_b');run('xai_off',['--override',off]);run('interrupted',['--stop-after','15']);run('resumed',['--resume',cls.root/'interrupted/checkpoint.pt'])
  cls.summaries={name:json.loads((cls.root/name/'summary.json').read_text()) for name in ['full_a','full_b','xai_off','resumed']}
  cls.evidence=ROOT/'evidence/operational_gates';cls.evidence.mkdir(exist_ok=True)
  for name in ['full_a','full_b','xai_off','resumed']:
   target=cls.evidence/name;target.mkdir(exist_ok=True)
   for filename in ['loss.txt','total_loss.txt','summary.json','manifest.json','metrics.jsonl','alarms.jsonl']:shutil.copyfile(cls.root/name/filename,target/filename)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def trace(self,name,file='loss.txt'):return (self.root/name/file).read_bytes()
 def test_two_seed_1234_runs_exact_and_strictly_decreasing(self):
  self.assertEqual(self.trace('full_a'),self.trace('full_b'));self.assertEqual(self.trace('full_a','total_loss.txt'),self.trace('full_b','total_loss.txt'))
  self.assertTrue(self.summaries['full_a']['every_train_step_decreased']);self.assertEqual(self.summaries['full_a']['nan_count'],0)
  self.assertEqual(self.summaries['full_a']['final_model_sha256'],self.summaries['full_b']['final_model_sha256'])
 def test_xai_zero_perturbation(self):
  self.assertEqual(self.trace('full_a'),self.trace('xai_off'));self.assertEqual(self.trace('full_a','total_loss.txt'),self.trace('xai_off','total_loss.txt'));self.assertEqual(self.summaries['full_a']['final_model_sha256'],self.summaries['xai_off']['final_model_sha256'])
 def test_checkpoint_resume_15_plus_15_exact(self):
  self.assertEqual(self.trace('full_a'),self.trace('resumed'));self.assertEqual(self.trace('full_a','total_loss.txt'),self.trace('resumed','total_loss.txt'));self.assertEqual(self.summaries['full_a']['final_model_sha256'],self.summaries['resumed']['final_model_sha256'])
  a=[json.loads(s) for s in (self.root/'full_a/metrics.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (self.root/'resumed/metrics.jsonl').read_text().splitlines()]
  for records in [a,b]:
   for record in records:record.pop('perf/step_seconds');record.pop('perf/tokens_per_second')
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
