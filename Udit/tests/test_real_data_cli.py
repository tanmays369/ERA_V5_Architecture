import json,os,pathlib,sys,tempfile,unittest
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
import train
TOKENIZER=pathlib.Path(os.environ.get('BRAHMIC_TOKENIZER','D:/github_repos/BrahmicTokenizer-131K/tokenizer.json'))
@unittest.skipUnless(TOKENIZER.exists(),'set BRAHMIC_TOKENIZER for CLI data-path integration')
class RealDataCLI(unittest.TestCase):
 def test_data_path_two_independent_initializations_match(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=pathlib.Path(tmp);source=root/'samples.jsonl';source.write_text('\n'.join(json.dumps({'text':f'A sample number {i} has deterministic tokens.\n'}) for i in range(12)),encoding='utf8')
   config=root/'tiny.yaml';text=(ROOT/'configs/tiny_moe.yaml').read_text();text=text.replace('warmup_steps: 3','warmup_steps: 1').replace('training:\n','training:\n  steps: 3 # source: test bounded real-data run\n');config.write_text(text)
   for name in ['a','b']:
    args=['train.py','--tokenizer',str(TOKENIZER),'--seed','1234','--data-path',str(source),'--override',str(config),'--out',str(root/name),'--device','cpu']
    with patch.object(sys,'argv',args):train.main()
   self.assertEqual((root/'a/loss.txt').read_bytes(),(root/'b/loss.txt').read_bytes())
   a=[json.loads(s) for s in (root/'a/metrics.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (root/'b/metrics.jsonl').read_text().splitlines()]
   self.assertEqual([r['data/batch_sha256'] for r in a],[r['data/batch_sha256'] for r in b])
   self.assertTrue(all('eval/perplexity' in r and 'eval/bpb' in r for r in a))
if __name__=='__main__':unittest.main()
