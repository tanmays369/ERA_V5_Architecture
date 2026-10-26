import os,pathlib,sys,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
from check_config import load_config
from model import Model
from optimizers import build_optimizers
TOKENIZER=pathlib.Path(os.environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER for tokenizer/model integration')
class ModelIntegration(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  torch.set_num_threads(1);torch.manual_seed(1234);cls.c=load_config(ROOT/'configs/config_v0.yaml',ROOT/'configs/tiny_moe.yaml')[0];cls.model=Model(cls.c,TOKENIZER)
 def test_telemetry_and_lens_do_not_change_loss(self):
  m=self.model;m.eval();x=torch.tensor([[130725,10,20]]);y=torch.tensor([[10,20,30]])
  with torch.no_grad():
   a=m(x,y,False,False);b=m(x,y,True,True)
  torch.testing.assert_close(a[0],b[0],rtol=0,atol=0)
  self.assertAlmostEqual(b[-1][-1]['logit_lens/ce'],float(b[0]/b[1]),places=6)
  self.assertIn('residual/input_output_cosine',b[-1][0]);self.assertAlmostEqual(b[-1][-1]['explain/logit_lens_kl'],0.,places=6)
 def test_native_probe_leaves_parameter_gradients_identical(self):
  m=self.model;m.train();x=torch.tensor([[130725,10,20]]);y=torch.tensor([[10,20,30]])
  m.zero_grad(set_to_none=True);a=m(x,y,False,False);(a[0]/a[1]+0.01*a[2]+0.001*a[3]).backward();expected={name:p.grad.clone() for name,p in m.named_parameters()}
  m.zero_grad(set_to_none=True);m.c['explainability']['sae_probe']['enabled']=True
  try:
   b=m(x,y,True,True);(b[0]/b[1]+0.01*b[2]+0.001*b[3]).backward()
   for name,p in m.named_parameters():torch.testing.assert_close(p.grad,expected[name],rtol=0,atol=0)
  finally:m.c['explainability']['sae_probe']['enabled']=False
 def test_four_roles_and_width_lr_hooks(self):
  import copy
  c=copy.deepcopy(self.c['optimizer']);c['mup_enabled']=True;c['mup_base_width']=48;c['router_lr_mult']=3.;c['ke2_lr_mult']=0.5
  opts,names=build_optimizers(self.model,c);self.assertEqual(set(names['roles']),{'dense_and_expert_2d','norms_and_biases_1d','moe_router_gate','ke2_kronecker_factors'})
  for opt in opts:
   for group in opt.param_groups:
    role=group['role']
    if role in ('norms_and_biases_1d','moe_router_gate'):self.assertEqual(group['weight_decay'],0.)
    if role=='moe_router_gate':self.assertEqual(group['lr_multiplier'],3.)
    if role=='ke2_kronecker_factors':self.assertEqual(group['lr_multiplier'],0.5)
    if role=='dense_and_expert_2d' and not group['parameter_name'].startswith('head.'):self.assertEqual(group['lr_multiplier'],0.5)
 def test_optimizer_roles(self):
  opts,names=build_optimizers(self.model,self.c['optimizer']);allnames=[name for k in ['muon','adamw_decay','adamw_no_decay'] for name in names[k]]
  self.assertEqual(len(allnames),len(set(allnames)));self.assertEqual(set(allnames),set(dict(self.model.named_parameters())))
  self.assertTrue(all(name in names['adamw_no_decay'] for name in allnames if name.startswith('embedding.')))
  self.assertIn('head.weight',names['adamw_decay']);self.assertTrue(all(not name.startswith(('embedding.','head.')) for name in names['muon']))
if __name__=='__main__':unittest.main()
