import copy,pathlib,sys,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
from check_config import load_config
from model import Attention,MoE
from optimizers import build_optimizers
C=load_config(ROOT/'configs/config_v0.yaml',ROOT/'configs/tiny_moe.yaml')[0]
class ModelTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(1234);torch.set_num_threads(1)
 def test_attention_causal(self):
  a=Attention(C);x=torch.randn(1,5,96);valid=torch.ones(1,5,dtype=torch.bool);z=x.clone();z[:,3:]+=100
  y,stats=a(x,valid,True);other,_=a(z,valid,False);torch.testing.assert_close(y[:,:3],other[:,:3],rtol=0,atol=0)
  self.assertTrue(all(0<=v<=__import__('math').log(5) for v in stats['attention/entropy_per_head']))
 def test_uniform_router_reference(self):
  m=MoE(C);m.router.weight.data.zero_();x=torch.randn(1,4,96);valid=torch.ones(1,4,dtype=torch.bool);out,aux,z,stats=m(x,valid)
  torch.testing.assert_close(aux,torch.tensor(1.));torch.testing.assert_close(z,torch.tensor(__import__('math').log(8)**2))
  expected=(m.experts[0](x)+m.experts[1](x))/2+m.shared[0](x)
  torch.testing.assert_close(out,expected);self.assertEqual(sum(stats['moe/expert_counts']),8)
  (out.square().mean()+aux+z).backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_padding_excluded_counts(self):
  m=MoE(C);_,_,_,stats=m(torch.randn(2,5,96),torch.tensor([[1,1,1,0,0],[1,0,0,0,0]],dtype=torch.bool));self.assertEqual(sum(stats['moe/expert_counts']),8)
 def test_latent_backward(self):
  c=copy.deepcopy(C);c['moe']['type']='latent';m=MoE(c);out,a,z,_=m(torch.randn(1,3,96),torch.ones(1,3,dtype=torch.bool));(out.square().mean()+a+z).backward();self.assertTrue(all(p.grad is not None for p in m.parameters()))
if __name__=='__main__':unittest.main()
