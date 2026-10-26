import copy,json,math,os,pathlib,sys,tempfile,unittest
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
from input_utils import LocalCorpus,CorpusWindows,DeterministicSampler,document_split,curriculum_at,sha256
from check_config import load_config
from model import ActivationHookManager,TopKSAEProbe
TOKENIZER=pathlib.Path(os.environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
C=load_config(ROOT/'configs/config_v0.yaml',ROOT/'configs/tiny_moe.yaml')[0]
class ContractTests(unittest.TestCase):
 def test_zero3_pin_false_rejected_and_true_contract_valid(self):
  text=(ROOT/'configs/tiny_moe.yaml').read_text().replace('zero_stage: 0','zero_stage: 3')
  with tempfile.TemporaryDirectory() as tmp:
   path=pathlib.Path(tmp)/'tiny.yaml';path.write_text(text);self.assertEqual(load_config(ROOT/'configs/config_v0.yaml',path)[1],[])
   path.write_text(text.replace('distributed:\n','distributed:\n  keep_router_in_fp32_under_zero: false # source: test broken pin\n'))
   self.assertTrue(any('FP32 router pinning' in e for e in load_config(ROOT/'configs/config_v0.yaml',path)[1]))
 def test_zero3_runtime_refuses_silent_fallback(self):
  from unittest.mock import patch
  import train
  with tempfile.TemporaryDirectory() as tmp:
   override=pathlib.Path(tmp)/'tiny.yaml';override.write_text((ROOT/'configs/tiny_moe.yaml').read_text().replace('zero_stage: 0','zero_stage: 3'))
   with patch.object(sys,'argv',['train.py','--override',str(override),'--tokenizer','unused','--out',str(pathlib.Path(tmp)/'run')]):
    with self.assertRaisesRegex(ValueError,'ZeRO-3 is a validated architecture contract only'):train.main()
 def test_curriculum_boundary(self):
  self.assertEqual([curriculum_at(C,s)[0] for s in [1,3,4,24,25,30]],['stage_1_warmup','stage_1_warmup','stage_2_main','stage_2_main','stage_3_anneal','stage_3_anneal'])
 def test_sampler_isolated_and_reproducible(self):
  state=torch.get_rng_state().clone();a=DeterministicSampler(7,1234);b=DeterministicSampler(7,1234)
  self.assertEqual(a.batch(20),b.batch(20));self.assertTrue(torch.equal(state,torch.get_rng_state()))
  self.assertNotEqual(DeterministicSampler(7,1234).batch(20),DeterministicSampler(7,1235).batch(20))
 def test_detached_hook_cannot_mutate_model_tensor(self):
  manager=ActivationHookManager(C['explainability']);manager.begin(True);x=torch.randn(1,3,96,requires_grad=True);before=x.detach().clone()
  manager.register(lambda point,layer,t:t.zero_());manager.capture('resid_mid',0,x,torch.ones(1,3,dtype=torch.bool))
  torch.testing.assert_close(x,before);snapshot=manager.snapshots[('resid_mid',0)];self.assertFalse(snapshot.requires_grad);torch.testing.assert_close(snapshot,before[0])
  x.square().sum().backward();torch.testing.assert_close(x.grad,2*x.detach())
 def test_probe_no_rng_effect_and_sparsity(self):
  state=torch.get_rng_state().clone();probe=TopKSAEProbe(96,32,4);self.assertTrue(torch.equal(state,torch.get_rng_state()))
  stats,active=probe(torch.ones(5,96));self.assertLessEqual(stats['feature_l0'],4);self.assertTrue(math.isfinite(stats['reconstruction_mse']));self.assertEqual(len(active),32)
 def test_dead_feature_alarm_requires_full_window(self):
  config=copy.deepcopy(C['explainability']);config['sae_probe']['dead_feature_window']=2;manager=ActivationHookManager(config)
  manager.register_sae_probe('resid_mid',0,lambda x:({'feature_l0':0.,'reconstruction_mse':1.},torch.zeros(4,dtype=torch.bool)))
  manager.begin(True);manager.capture('resid_mid',0,torch.ones(1,2,96),torch.ones(1,2,dtype=torch.bool));self.assertFalse(manager.events)
  manager.begin(True);manager.capture('resid_mid',0,torch.ones(1,2,96),torch.ones(1,2,dtype=torch.bool));self.assertEqual(manager.events[0]['name'],'alarm/sae_dead_features')
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER for corpus ingestion tests')
class DataTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.path=pathlib.Path(self.tmp.name);self.corpora=[]
 def tearDown(self):
  for c in self.corpora:c.close()
  self.tmp.cleanup()
 def corpus(self,name):
  c=LocalCorpus(TOKENIZER,self.path/name);self.corpora.append(c);return c
 def test_jsonl_txt_and_chunk_byte_accounting(self):
  text='A repeated sample with Unicode: नमस्ते.\n'*20;src=self.path/'sample.jsonl';src.write_text(json.dumps({'text':text})+'\n'+json.dumps({'text':'Another document.'}),encoding='utf8')
  c=self.corpus('raw');c.load(src);c.finish();windows=CorpusWindows(c,[0],16);total=0;tokens=0
  for i in range(len(windows)):
   x,y,nbytes=windows.batch([i],'cpu');total+=nbytes;tokens+=int(y.ne(-100).sum());self.assertEqual(tuple(x.shape),(1,16))
  self.assertEqual(total,len(text.encode('utf8')));self.assertEqual(tokens,c.docs[0][1]);self.assertGreater(len(windows),1)
  txt=self.path/'sample.txt';txt.write_text('First line.\nSecond line.\n',encoding='utf8');other=self.corpus('txt');other.load(txt);other.finish();self.assertEqual(sum(d[3] for d in other.docs),len(txt.read_bytes()))
 def test_bin_and_pt_equal_raw(self):
  c=self.corpus('original');c.add_text('A first sample.');c.add_text('A second sample.');c.finish();ids=[]
  for offset,length,_,_ in c.docs:ids.extend(c.tokens[offset:offset+length].tolist());ids.append(36)
  binary=self.path/'shard.bin';np.asarray(ids,dtype='<u4').tofile(binary);pathlib.Path(str(binary)+'.meta.json').write_text(json.dumps({'dtype':'uint32','tokenizer_sha256':sha256(TOKENIZER)}))
  a=self.corpus('bin');a.load(binary);a.finish();self.assertEqual(a.docs,c.docs)
  pt=self.path/'shard.pt';torch.save({'input_ids':torch.tensor(ids,dtype=torch.int64),'tokenizer_sha256':sha256(TOKENIZER)},pt)
  b=self.corpus('pt');b.load(pt);b.finish();self.assertEqual(b.docs,c.docs)
 def test_unpinned_or_float_shard_rejected(self):
  file=self.path/'bad.bin';np.asarray([10,20],dtype='<u4').tofile(file)
  with self.assertRaisesRegex(ValueError,'tokenizer_sha256'):self.corpus('bad').load(file)
  pt=self.path/'float.pt';torch.save({'input_ids':torch.tensor([10.,20.]),'tokenizer_sha256':sha256(TOKENIZER)},pt)
  with self.assertRaisesRegex(ValueError,'Floating'):self.corpus('float').load(pt)
 def test_duplicate_documents_not_split_between_partitions(self):
  c=self.corpus('dups');c.add_text('Repeated text.');c.add_text('Repeated text.');c.add_text('Other text.');c.finish();a,b=document_split(c)
  self.assertFalse({c.docs[i][2] for i in a}&{c.docs[i][2] for i in b})
if __name__=='__main__':unittest.main()
