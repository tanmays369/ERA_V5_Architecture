import copy,json,math,os,pathlib,sys,tempfile,unittest
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
from check_config import load_config
from input_utils import CurriculumEngine,LocalCorpus,CorpusWindows,MixtureSampler,token_budget_batch
from model import ActivationHookManager,MoE,Block,Attention,KE2Adapter,boundary_metadata,Model
from optimizers import WSDScheduler,build_optimizers
C=load_config(ROOT/'configs/config_v0.yaml',ROOT/'configs/tiny_moe.yaml')[0]
TOKENIZER=pathlib.Path(os.environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
class FrontierTests(unittest.TestCase):
 def setUp(self):torch.manual_seed(1234);torch.set_num_threads(1)
 def test_smooth_transition_and_floor(self):
  c=copy.deepcopy(C);cur=c['data_curriculum'];cur['domain_labels']=['a','b'];cur['reference_losses']=[1.,1.]
  cur['stage_1_warmup']['mixture_weights']=[1.,0.];cur['stage_2_main']['mixture_weights']=[0.,1.];cur['stage_3_anneal']['mixture_weights']=[0.,1.];cur['stage_2_main']['router_temperature']=2.
  engine=CurriculumEngine(c);a=engine.at(3);b=engine.at(4);mid=engine.at(5);end=engine.at(6)
  self.assertEqual(a['mixture_weights'],b['mixture_weights']);self.assertEqual(b['transition_progress'],0);self.assertAlmostEqual(mid['router_temperature'],1.5);self.assertEqual(end['router_temperature'],2.)
  self.assertTrue(all(w>=0.05 for w in end['mixture_weights']))
 def test_excess_loss_hook_updates_and_restores(self):
  c=copy.deepcopy(C);cur=c['data_curriculum'];cur['weighting_mode']='online_excess_loss';cur['domain_labels']=['a','b'];cur['reference_losses']=[1.,1.]
  for name in ['stage_1_warmup','stage_2_main','stage_3_anneal']:cur[name]['mixture_weights']=[0.5,0.5]
  engine=CurriculumEngine(c);engine.observe_excess([3.,1.],[1.,1.],2);self.assertGreater(engine.at(3)['mixture_weights'][0],0.5)
  other=CurriculumEngine(c);other.load_state_dict(engine.state_dict());self.assertEqual(engine.at(4),other.at(4))
 def test_bias_selection_uses_unbiased_gates_and_one_update(self):
  c=copy.deepcopy(C);c['moe']['router_balance_mode']='aux_loss_free_bias';m=MoE(c);m.router.weight.data.zero_();m.router_bias[7]=1.
  x=torch.randn(1,4,96);valid=torch.ones(1,4,dtype=torch.bool);out,aux,z,stats=m(x,valid)
  expected=(m.experts[7](x)+m.experts[0](x))/2+m.shared[0](x);torch.testing.assert_close(out,expected);self.assertEqual(float(aux),0.)
  before=m.router_bias.clone();counts=m.pending_counts.clone();m.finish_update();torch.testing.assert_close(m.router_bias,before+0.001*torch.sign(counts.mean()-counts));self.assertEqual(float(m.pending_counts.sum()),0)
  m.eval();before=m.router_bias.clone();m(x,valid);self.assertEqual(float(m.pending_counts.sum()),0);torch.testing.assert_close(before,m.router_bias)
 def test_padding_capacity_and_shared_path(self):
  c=copy.deepcopy(C);c['moe']['dropless_moe']=False;c['moe']['train_capacity_factor']=0.1;m=MoE(c);m.router.weight.data.zero_();valid=torch.tensor([[1,1,1,0]],dtype=torch.bool)
  out,a,z,stats=m(torch.randn(1,4,96),valid);self.assertLess(sum(stats['moe/expert_counts']),6);self.assertGreater(stats['moe/dropped_token_ratio'],0);self.assertEqual(float(out[0,3].abs().sum()),0)
 def test_dense_prefix_has_no_router(self):
  c=copy.deepcopy(C);c['moe']['first_k_dense_replace']=1;a=Block(c,0);b=Block(c,1);self.assertFalse(hasattr(a.moe,'router'));self.assertTrue(hasattr(b.moe,'router'))
 def test_boundary_positions_and_cross_document_mask(self):
  ids=torch.tensor([[130725,10,36,20,21]]);positions,documents=boundary_metadata(ids,36,130725);self.assertEqual(positions.tolist(),[[0,1,2,0,1]])
  attn=Attention(C);x=torch.randn(1,5,96);other=x.clone();other[:,:3]+=100;valid=torch.ones(1,5,dtype=torch.bool)
  y,_=attn(x,valid,True,positions,documents);z,_=attn(other,valid,True,positions,documents);torch.testing.assert_close(y[:,3:],z[:,3:],atol=0,rtol=0)
 def test_softcap_is_smooth_bounded_and_qk_toggle(self):
  c=copy.deepcopy(C);c['attention']['attn_logit_softcapping']=10.;c['attention']['use_qk_norm']=False;a=Attention(c);x=torch.randn(1,4,96)*100;y,_=a(x,torch.ones(1,4,dtype=torch.bool),True);self.assertTrue(torch.isfinite(y).all())
 def test_ke_adapter_variance_and_gradient(self):
  c=copy.deepcopy(C);base=torch.nn.Embedding(8,48);adapter=KE2Adapter(base,48,96,c);output=adapter(torch.tensor([[1,2,3]]));self.assertEqual(tuple(output.shape),(1,3,96));output.square().sum().backward();self.assertTrue(torch.isfinite(base.weight.grad).all())
 def test_factorized_tied_output_formula_and_gradients(self):
  import types
  c=copy.deepcopy(C);c['tokenizer']['vocab_size']=8;c['output']['factorized_chunk_vocab']=3
  embedding=torch.nn.Embedding(8,6);obj=types.SimpleNamespace(c=c,head=None,embedding=embedding);x=torch.randn(2,6,requires_grad=True)
  for mode in ['tied_dense','factorized_ke2_head']:
   c['embeddings_ke2']['lm_head_mode']=mode;actual=Model.project_logits(obj,x);expected=torch.nn.functional.linear(x,embedding.weight)
   torch.testing.assert_close(actual,expected)
   grads=torch.autograd.grad(actual.square().sum(),(x,embedding.weight),retain_graph=True);ref=torch.autograd.grad(expected.square().sum(),(x,embedding.weight));torch.testing.assert_close(grads[0],ref[0]);torch.testing.assert_close(grads[1],ref[1])
 def test_mup_attention_formula(self):
  c=copy.deepcopy(C);c['optimizer']['mup_enabled']=True;a=Attention(c);x=torch.randn(1,3,96);valid=torch.ones(1,3,dtype=torch.bool)
  q=a.rope(a.qnorm(a.q(x).view(1,3,3,32).transpose(1,2)));k=a.rope(a.knorm(a.k(x).view(1,3,1,32).transpose(1,2))).repeat_interleave(3,1);v=a.v(x).view(1,3,1,32).transpose(1,2).repeat_interleave(3,1)
  logits=(q@k.transpose(-1,-2))/32;prob=logits.masked_fill(~torch.ones(3,3,dtype=torch.bool).tril(),float('-inf')).softmax(-1);expected=a.out((prob@v).transpose(1,2).reshape(1,3,96));actual,_=a(x,valid,False);torch.testing.assert_close(actual,expected)
 def test_callback_rng_isolation(self):
  import random,numpy as np
  manager=ActivationHookManager(C['explainability']);manager.begin(True)
  def callback(*args):torch.rand(7);random.random();np.random.random()
  manager.register(callback);x=torch.ones(1,3,96);torch_state=torch.get_rng_state().clone();python_state=random.getstate();numpy_state=np.random.get_state()
  manager.capture('resid_mid',0,x,torch.ones(1,3,dtype=torch.bool));self.assertTrue(torch.equal(torch_state,torch.get_rng_state()));self.assertEqual(python_state,random.getstate());self.assertTrue(np.array_equal(numpy_state[1],np.random.get_state()[1]))
 def test_wsd_coupled_to_anneal(self):
  scheduler=WSDScheduler([],C);self.assertAlmostEqual(scheduler.at(3),1.);self.assertEqual(scheduler.at(24),1.);self.assertGreater(scheduler.at(25),scheduler.at(30));self.assertAlmostEqual(scheduler.at(30),0.1)
 def test_g7_open_sizes_does_not_waive_sources(self):
  self.assertEqual(load_config(ROOT/'configs/config_v0.yaml',allow_open=True)[1],[])
  text=(ROOT/'configs/config_v0.yaml').read_text().replace('compute_dtype: "float32"','compute_dtype: "fp8"')
  with tempfile.TemporaryDirectory() as tmp:
   path=pathlib.Path(tmp)/'bad.yaml';path.write_text(text);self.assertTrue(load_config(path,allow_open=True)[1])
   path.write_text(text.replace(' # source:',' # not_source:'));self.assertTrue(load_config(path,allow_open=True)[1])
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER')
class BudgetTests(unittest.TestCase):
 def test_exact_valid_token_budget_and_sampler_resume(self):
  with tempfile.TemporaryDirectory() as tmp:
   c=LocalCorpus(TOKENIZER,tmp)
   try:
    c.add_text('A small document contains several tokens and repeats. '*6);c.add_text('Another independent document. '*8);c.finish();accum=[]
    for length in [4,8,16]:
     w=CorpusWindows(c,[0,1],length);s=MixtureSampler(w,['slice_0'],1234);seen=0;micro=0
     while seen<24:
      x,y,nbytes=token_budget_batch(w,s.batch(2,[1.]),24-seen,'cpu');seen+=int(y.ne(-100).sum());micro+=1
     self.assertEqual(seen,24);accum.append(micro);state=s.state_dict();expected=s.batch(5,[1.]);other=MixtureSampler(w,['slice_0'],1234);other.load_state_dict(state);self.assertEqual(expected,other.batch(5,[1.]))
    self.assertGreater(accum[0],accum[-1])
   finally:c.close()

 def test_attention_mechanisms_and_router_scores_forward_backward_deterministic(self):
  for mechanism in ['gqa','mha','mqa','mla','sliding_window_hybrid','diff_attn']:
   for scoring in ['softmax','sigmoid']:
    with self.subTest(mechanism=mechanism,scoring=scoring):
     c=copy.deepcopy(C);c['attention']['mechanism']=mechanism;c['moe']['scoring_func']=scoring
     c['attention']['v_head_dim']=24
     if mechanism=='mla':
      c['attention'].update(kv_lora_rank=16,q_lora_rank=16,qk_nope_head_dim=24,qk_rope_head_dim=8)
     torch.manual_seed(1234);m=Block(c,1);m.train();x=torch.randn(1,8,96,requires_grad=True);valid=torch.ones(1,8,dtype=torch.bool);pos=torch.arange(8)[None,:];docs=torch.zeros(1,8,dtype=torch.long)
     a=m(x,valid,False,None,1,pos,docs);la=a[0].square().mean()+a[1]*c['moe']['aux_coefficient']+a[2]*c['moe']['router_z_coefficient'];ga=torch.autograd.grad(la,(x,*tuple(m.parameters())),allow_unused=True)
     b=m(x,valid,False,None,1,pos,docs);lb=b[0].square().mean()+b[1]*c['moe']['aux_coefficient']+b[2]*c['moe']['router_z_coefficient'];gb=torch.autograd.grad(lb,(x,*tuple(m.parameters())),allow_unused=True)
     self.assertTrue(torch.isfinite(la));self.assertTrue(all(g is None or torch.isfinite(g).all() for g in ga))
     torch.testing.assert_close(la,lb,rtol=0,atol=0)
     for x,y in zip(ga,gb):
      if x is not None:torch.testing.assert_close(x,y,rtol=0,atol=0)
 def test_attention_incremental_cache_matches_full_sequence(self):
  for mechanism in ['gqa','mla','sliding_window_hybrid','diff_attn']:
   with self.subTest(mechanism=mechanism):
    c=copy.deepcopy(C);c['attention']['mechanism']=mechanism;c['attention']['v_head_dim']=24
    if mechanism=='mla':c['attention'].update(kv_lora_rank=16,q_lora_rank=16,qk_nope_head_dim=24,qk_rope_head_dim=8)
    torch.manual_seed(1234);a=Attention(c).eval();x=torch.randn(1,7,96);valid=torch.ones(1,7,dtype=torch.bool);pos=torch.arange(7)[None,:];docs=torch.zeros(1,7,dtype=torch.long)
    with torch.no_grad():
     expected,_=a(x,valid,False,pos,docs);cache=None;chunks=[]
     for i in range(x.shape[1]):
      out,_,cache=a(x[:,i:i+1],valid[:,i:i+1],False,pos[:,i:i+1],docs[:,i:i+1],cache=cache,use_cache=True);chunks.append(out)
     actual=torch.cat(chunks,dim=1)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
 def test_attention_geometry_config_contracts(self):
  from check_config import validate_completion
  for mechanism in ['gqa','mha','mqa','mla','sliding_window_hybrid','diff_attn']:
   c=copy.deepcopy(C);c['attention']['mechanism']=mechanism
   if mechanism=='mha':c['attention']['kv_heads']=c['attention']['query_heads']
   if mechanism=='mla':c['attention'].update(kv_lora_rank=16,q_lora_rank=16,qk_nope_head_dim=24,qk_rope_head_dim=8)
   self.assertEqual(validate_completion(c),[],mechanism)
  c=copy.deepcopy(C);c['attention']['mechanism']='mla';c['attention']['qk_head_dim']=31
  self.assertTrue(any('MLA QK' in error for error in validate_completion(c)))

 def test_model_grouped_kv_cache_decode_matches_prefix_recompute(self):
  class TinyEmbedding(torch.nn.Module):
   def __init__(self):super().__init__();self.table=torch.nn.Embedding(128,96)
   def forward(self,ids):return self.table(ids)
  for mechanism in ['gqa','mla','sliding_window_hybrid','diff_attn']:
   with self.subTest(mechanism=mechanism):
    c=copy.deepcopy(C);c['tokenizer'].update(vocab_size=128,padded_vocab_size=128,bos_id=0,eos_id=1,pad_id=127);c['attention'].update(mechanism=mechanism,kv_sharing_group_size=2,v_head_dim=24)
    if mechanism=='mla':c['attention'].update(kv_lora_rank=16,q_lora_rank=16,qk_nope_head_dim=24,qk_rope_head_dim=8)
    torch.manual_seed(4321);model=Model(c,TOKENIZER,embedding_provider=TinyEmbedding()).eval();ids=torch.tensor([[2,3,4,5,6]])
    positions=torch.arange(ids.shape[1])[None,:];documents=torch.ones_like(ids)
    with torch.no_grad():
     full_logits,_=model(ids,None,position_ids=positions,document_ids=documents,use_cache=True)
     cache=None
     for i in range(ids.shape[1]):
      kwargs={'position_ids':positions[:,i:i+1],'document_ids':documents[:,i:i+1]}
      if cache is None:logits,cache=model(ids[:,i:i+1],None,use_cache=True,**kwargs)
      else:logits,cache=model(ids[:,i:i+1],None,past_key_values=cache,use_cache=True,**kwargs)
     torch.testing.assert_close(logits,full_logits,rtol=1e-5,atol=1e-5)


if __name__=='__main__':unittest.main()
