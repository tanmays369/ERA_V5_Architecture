"""Exercise actual train.py fail-fast paths; requires pinned local tokenizer."""
import json,pathlib,sys,tempfile,unittest
from unittest.mock import patch
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
import train,model,optimizers
TOKENIZER=pathlib.Path(__import__('os').environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER for full alarm integration tests')
class AlarmIntegration(unittest.TestCase):
 def run_fault(self,kind):
  original_forward=model.Model.forward;original_block=model.Block.forward;original_init=model.Model.__init__;original_step=optimizers.ReferenceMuon.step
  def forward(instance,*args,**kwargs):
   result=original_forward(instance,*args,**kwargs)
   if instance.training and kind=='loss':return (result[0]*float('nan'),)+result[1:]
   return result
  def block(instance,*args,**kwargs):
   if instance.training and kind=='activation':raise FloatingPointError('alarm/nonfinite_activation')
   return original_block(instance,*args,**kwargs)
  def init(instance,*args,**kwargs):
   original_init(instance,*args,**kwargs)
   if kind=='gradient':instance.head.weight.register_hook(lambda grad:grad*float('nan'))
  def step(instance,*args,**kwargs):
   result=original_step(instance,*args,**kwargs)
   if kind=='parameter':
    with torch.no_grad():instance.param_groups[0]['params'][0].fill_(float('nan'))
   return result
  with tempfile.TemporaryDirectory() as tmp:
   out=pathlib.Path(tmp)/'run';args=['train.py','--seed','1234','--tokenizer',str(TOKENIZER),'--out',str(out),'--device','cpu']
   with patch.object(sys,'argv',args),patch.object(model.Model,'forward',forward),patch.object(model.Block,'forward',block),patch.object(model.Model,'__init__',init),patch.object(optimizers.ReferenceMuon,'step',step):
    with self.assertRaises(FloatingPointError):train.main()
   events=[json.loads(line) for line in (out/'alarms.jsonl').read_text().splitlines()];self.assertEqual(events[-1]['name'],'alarm/nonfinite_'+kind)
 def test_loss_alarm(self):self.run_fault('loss')
 def test_activation_alarm(self):self.run_fault('activation')
 def test_gradient_alarm(self):self.run_fault('gradient')
 def test_parameter_alarm(self):self.run_fault('parameter')
if __name__=='__main__':unittest.main()
