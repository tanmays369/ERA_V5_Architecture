"""Verify independent-run traces plus finite telemetry and probe reduction."""
import argparse,json,math,pathlib

def main():
 p=argparse.ArgumentParser();p.add_argument('run1',type=pathlib.Path);p.add_argument('run2',type=pathlib.Path);a=p.parse_args()
 if (a.run1/'loss.txt').read_bytes()!=(a.run2/'loss.txt').read_bytes():raise SystemExit('FAIL: loss trace differs')
 for root in [a.run1,a.run2]:
  summary=json.loads((root/'summary.json').read_text());assert summary['status']=='PASS';assert summary['nan_count']==0
  assert summary['final_probe']['ce']<summary['initial_probe']['ce']
  def finite(value):
   if isinstance(value,dict):return all(finite(v) for v in value.values())
   if isinstance(value,list):return all(finite(v) for v in value)
   return not isinstance(value,float) or math.isfinite(value)
  records=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()];assert records and finite(records)
  for rec in records:
   assert math.isclose(rec['train/token_perplexity'],math.exp(rec['train/ce']),rel_tol=1e-12)
   assert math.isclose(rec['train/bpb'],rec['train/ce']*rec['train/valid_tokens']/(math.log(2)*rec['train/raw_utf8_bytes']),rel_tol=1e-6)
  assert not (root/'alarms.jsonl').read_text().strip()
  assert len(records)==json.loads((root/'manifest.json').read_text())['config']['training']['steps']
  if json.loads((root/'manifest.json').read_text())['data_path']=='fixture':assert summary['every_train_step_decreased']
  for rec in records:
   if 'eval/perplexity' in rec:assert math.isclose(rec['eval/perplexity'],math.exp(rec['eval/ce']),rel_tol=1e-12)
 manifests=[json.loads((r/'manifest.json').read_text()) for r in [a.run1,a.run2]]
 for key in ['seed','config','source_hashes','tokenizer_sha256','torch','cuda','device','train_text_sha256','validation_text_sha256']:assert manifests[0][key]==manifests[1][key],key
 deterministic=[]
 for root in [a.run1,a.run2]:
  records=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
  for rec in records:rec.pop('perf/step_seconds');rec.pop('perf/tokens_per_second')
  deterministic.append(records)
 assert deterministic[0]==deterministic[1], 'Non-timing metrics or batches differ'
 print('PASS: empty byte diff; finite metrics; zero alarms; fixed-probe CE decreases; provenance matches')
if __name__=='__main__':main()
