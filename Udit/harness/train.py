"""Single-process deterministic research trainer with exact token budgets and resume."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,math,pathlib,random,sys,time
from check_config import ROOT,load_config
from input_utils import sha256,LocalCorpus,CorpusWindows,document_split,CurriculumEngine,MixtureSampler,token_budget_batch


def aggregate_layer_metrics(batches,k):
 total=sum(count for count,_ in batches);result=[]
 for layer in range(len(batches[0][1])):
  row={};keys=set().union(*(stats[layer] for _,stats in batches))
  for key in keys:
   values=[(count,stats[layer][key]) for count,stats in batches if key in stats[layer]];denom=sum(n for n,_ in values);sample=values[0][1]
   if key=='moe/expert_counts':row[key]=[sum(value[i] for _,value in values) for i in range(len(sample))]
   elif type(sample) in (int,float):row[key]=sum(n*value for n,value in values)/denom
   elif isinstance(sample,list) and sample and all(type(x) in (int,float) for x in sample):row[key]=[sum(n*value[i] for n,value in values)/denom for i in range(len(sample))]
   else:row[key]=values[-1][1]
  if 'moe/expert_counts' in row:
   counts=row['moe/expert_counts'];assigned=sum(counts);e=len(counts);fraction=[v/assigned if assigned else 0 for v in counts];entropy=-sum(p*math.log(p) for p in fraction if p>0)/math.log(e) if e>1 else 0.
   row['moe/utilization_entropy']=entropy;row['moe/max_load_violation']=max(counts)/(k*total/e)-1;row['moe/dead_expert_count']=sum(v==0 for v in counts)
   for a,b in [('moe/utilization_entropy','explain/expert_utilization_entropy'),('moe/max_load_violation','explain/max_load_violation'),('moe/dead_expert_count','explain/dead_expert_count')]:
    if b in row:row[b]=row[a]
  result.append(row)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default=ROOT/'configs/config_v0.yaml');p.add_argument('--override',default=ROOT/'configs/tiny_moe.yaml');p.add_argument('--seed',type=int);p.add_argument('--tokenizer',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--out',required=True);p.add_argument('--data-path','--data',dest='data_path');p.add_argument('--validation-data-path','--validation-data',dest='validation_data_path');p.add_argument('--slice-documents',type=int);p.add_argument('--resume');p.add_argument('--stop-after',type=int);args=p.parse_args()
 c,errors=load_config(args.config,args.override)
 if errors:raise ValueError('\n'.join(errors))
 args.seed=c['training']['seed'] if args.seed is None else args.seed;c['training']['seed']=args.seed
 if c['distributed']['zero_stage']!=0:raise ValueError('ZeRO-3 is a validated architecture contract only; actual runtime integration is pending')
 if any(c['distributed'][key]!=1 for key in ['dp_size','ep_size','tp_size','cp_size']):raise ValueError('4D parallelism is a schema contract; this runtime requires every size equal to one')
 if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('Single-process reference; no DDP engine claimed')
 if sha256(args.tokenizer)!=c['tokenizer']['sha256']:raise ValueError('Tokenizer fingerprint mismatch')
 import numpy as np
 import torch
 from model import Model,MoE
 from optimizers import build_optimizers,WSDScheduler
 random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(1)
 if args.device=='cuda':torch.cuda.manual_seed_all(args.seed)
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=c['precision']['allow_tf32'];torch.backends.cudnn.allow_tf32=c['precision']['allow_tf32']
 device=torch.device(args.device);out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=False)
 corpus=LocalCorpus(args.tokenizer,out/'data_cache');validation_corpus=None;corpus.labels=c['data_curriculum']['domain_labels'];corpus.current_slice=corpus.labels[0]
 selected_path=args.data_path or c['data']['dataset_path'];validation_path=args.validation_data_path or c['data']['validation_dataset_path'];limit=args.slice_documents if args.slice_documents is not None else c['data']['slice_documents']
 if limit<2:raise ValueError('Slice needs >=2 documents')
 if selected_path=='fixture':
  for text in ['The small model learns from a fixed sequence.\n','A second example keeps this test reproducible.\n','The model predicts the next token.\n','This example is reserved for validation.\n']:corpus.add_text(text)
  corpus.finish();train_docs=[0,1];val_docs=[2,3]
 else:
  corpus.load(selected_path,limit if c['data']['mode']=='deterministic_slice' else 0);corpus.finish()
  if validation_path!='auto_split':
   validation_corpus=LocalCorpus(args.tokenizer,out/'validation_cache');validation_corpus.labels=corpus.labels;validation_corpus.current_slice=corpus.labels[0];validation_corpus.load(validation_path,limit if c['data']['mode']=='deterministic_slice' else 0);validation_corpus.finish();train_docs=list(range(len(corpus.docs)));val_docs=list(range(len(validation_corpus.docs)))
   if {d[2] for d in corpus.docs}&{d[2] for d in validation_corpus.docs}:raise ValueError('Exact train/validation overlap')
  else:train_docs,val_docs=document_split(corpus)
 batch_size=c['training']['batch_size'];probe=CorpusWindows(corpus,train_docs,c['training']['sequence_length']);val=CorpusWindows(validation_corpus or corpus,val_docs,c['training']['sequence_length']);probe_indices=list(range(min(c['evaluation']['eval_batches']*batch_size,len(probe))));val_indices=list(range(min(c['evaluation']['eval_batches']*batch_size,len(val))))
 model=Model(c,args.tokenizer).to(device);opts,groups=build_optimizers(model,c['optimizer']);scheduler=WSDScheduler(opts,c);engine=CurriculumEngine(c)
 hashes={str(f.relative_to(ROOT)):sha256(f) for f in ROOT.rglob('*') if f.is_file() and f.suffix in ('.py','.yaml','.md','.json') and 'evidence' not in f.parts}
 manifest=dict(seed=args.seed,config=c,source_hashes=hashes,tokenizer_sha256=sha256(args.tokenizer),torch=str(torch.__version__),python=sys.version,cuda=torch.version.cuda,device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,parameters=sum(p.numel() for p in model.parameters()),optimizer_groups=groups,data_sources=corpus.sources,validation_sources=validation_corpus.sources if validation_corpus else [],data_path=selected_path,data_mode=c['data']['mode'],train_text_sha256=hashlib.sha256(json.dumps([corpus.docs[i][2] for i in train_docs]).encode()).hexdigest(),validation_text_sha256=hashlib.sha256(json.dumps([(validation_corpus or corpus).docs[i][2] for i in val_docs]).encode()).hexdigest(),bpb='Text-only NLL / ln(2) / exact scored bytes; structural EOS/PAD excluded',limitations=['Single process FP32','Dense expert reference execution','G4 provider not delivered','muP-inspired hooks need coordinate/transfer validation'])
 probe_config=c['explainability']['sae_probe'];manifest['sae_checkpoint_sha256']=sha256(probe_config['weights_path']) if probe_config['enabled'] and probe_config['kind']=='trained_topk' else None
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
 files={name:(out/name).open('w',encoding='utf8') for name in ['metrics.jsonl','loss.txt','total_loss.txt','alarms.jsonl']}
 def alarm(name,step):files['alarms.jsonl'].write(json.dumps(dict(name=name,step=step))+'\n');files['alarms.jsonl'].flush()
 @torch.no_grad()
 def evaluate(windows,indices):
  model.eval();total=0.;tokens=0;bytes_=0;old=[block.attn.theta for block in model.blocks];temperatures=[(module,module.temperature) for module in model.modules() if isinstance(module,MoE)]
  for module,_ in temperatures:module.temperature=c['data_curriculum']['stage_3_anneal']['router_temperature'] if c['data_curriculum']['enabled'] else 1.
  for block in model.blocks:block.attn.theta=c['data_curriculum']['stage_3_anneal']['rope_theta'] if c['data_curriculum']['enabled'] else c['attention']['rope_theta']
  try:
   for start in range(0,len(indices),batch_size):
    x,y,nbytes=windows.batch(indices[start:start+batch_size],device);nll,n,*_=model(x,y);total+=float(nll);tokens+=int(n);bytes_+=nbytes
  finally:
   for block,theta in zip(model.blocks,old):block.attn.theta=theta
   for module,temperature in temperatures:module.temperature=temperature
   model.train()
  if not tokens:raise ValueError('Empty evaluation slice')
  return dict(ce=total/tokens,bpb=total/(math.log(2)*bytes_),perplexity=math.exp(total/tokens))
 history=[];records=[];active_stage=None;train=None;sampler=None;start_step=1
 try:
  try:initial=evaluate(probe,probe_indices);initial_val=evaluate(val,val_indices)
  except FloatingPointError:alarm('alarm/nonfinite_activation',0);raise
  if args.resume:
   state=torch.load(args.resume,map_location='cpu',weights_only=True)
   if not state.get('resume_capable',False):raise ValueError('Checkpoint is an export without complete optimizer/RNG/dataloader state')
   for key in ['config','source_hashes','tokenizer_sha256','torch','cuda','device','gpu','train_text_sha256','validation_text_sha256','sae_checkpoint_sha256']:
    if state['manifest'][key]!=manifest[key]:raise ValueError('Resume provenance mismatch: '+key)
   model.load_state_dict(state['model'],strict=True)
   for opt,saved in zip(opts,state['optimizers']):opt.load_state_dict(saved)
   scheduler.load_state_dict(state['scheduler']);engine.load_state_dict(state['curriculum']);active_stage=state['active_stage'];stage=engine.at(state['step']);train=CorpusWindows(corpus,train_docs,stage['seq_len']);sampler=MixtureSampler(train,corpus.labels,args.seed+(['stage_1_warmup','stage_2_main','stage_3_anneal'].index(stage['stage']) if stage['stage']!='disabled' else 0)*c['data_curriculum']['stage_rng_seed_stride']);sampler.load_state_dict(state['sampler'])
   initial=state['initial'];initial_val=state['initial_validation'];history=state['history'];records=state['records'];model.hooks.histories=state['probe_histories']
   for record in records:
    files['metrics.jsonl'].write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');files['loss.txt'].write(f"{record['step']:06d} {record['loss/ce']:.17g}\n");files['total_loss.txt'].write(f"{record['step']:06d} {record['loss/total']:.17g}\n")
   random.setstate(state['python_rng']);rng=state['numpy_rng'];np.random.set_state((rng[0],rng[1].numpy().astype(np.uint32),rng[2],rng[3],rng[4]));torch.set_rng_state(state['cpu_rng'])
   if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
   start_step=state['step']+1
  def save(step):
   rng=np.random.get_state();state=dict(version=1,step=step,manifest=manifest,model=model.state_dict(),optimizers=[opt.state_dict() for opt in opts],scheduler=scheduler.state_dict(),curriculum=engine.state_dict(),active_stage=active_stage,sampler=sampler.state_dict(),initial=initial,initial_validation=initial_val,history=history,records=records,probe_histories=model.hooks.histories,python_rng=random.getstate(),numpy_rng=(rng[0],torch.tensor(rng[1].astype(np.int64)),rng[2],rng[3],rng[4]),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
   state['resume_capable']=c['checkpointing']['save_optimizer_state'] and c['checkpointing']['save_rng_and_dataloader_state']
   if not c['checkpointing']['save_optimizer_state']:state.pop('optimizers')
   if not c['checkpointing']['save_rng_and_dataloader_state']:
    for key in ['python_rng','numpy_rng','cpu_rng','cuda_rng','sampler']:state.pop(key)
   temporary=out/'checkpoint.tmp';target=out/f'checkpoint_step_{step:06d}.pt';torch.save(state,temporary);os.replace(temporary,target)
   latest=out/'checkpoint_latest.tmp'
   try:os.link(target,latest)
   except OSError:
    import shutil
    shutil.copyfile(target,latest)
   os.replace(latest,out/'checkpoint.pt')
   older=sorted(out.glob('checkpoint_step_*.pt'))[:-c['checkpointing']['keep_last_n']]
   for path in older:
    if path.parent.resolve()!=out.resolve():raise ValueError('Checkpoint retention path escaped output directory')
    path.unlink()
  stop=min(args.stop_after if args.stop_after is not None else c['training']['steps'],c['training']['steps']);clock=time.perf_counter()
  if stop<1 or stop<start_step-1:raise ValueError('stop-after must be positive and not precede resumed update')
  for step in range(start_step,stop+1):
   if device.type=='cuda':torch.cuda.synchronize()
   tick=time.perf_counter();stage=engine.at(step);stage_name=stage['stage']
   if stage_name!=active_stage:
    active_stage=stage_name;train=CorpusWindows(corpus,train_docs,stage['seq_len']);sampler=MixtureSampler(train,corpus.labels,args.seed+(['stage_1_warmup','stage_2_main','stage_3_anneal'].index(stage['stage']) if stage['stage']!='disabled' else 0)*c['data_curriculum']['stage_rng_seed_stride'])
   for block in model.blocks:block.attn.theta=stage['rope_theta']
   for module in model.modules():
    if isinstance(module,MoE):module.temperature=stage['router_temperature']
   telemetry=c['explainability']['enabled'] and (step==1 or step%c['explainability']['tap_every_steps']==0);lens=c['explainability']['enabled'] and c['explainability']['enable_logit_lens'] and (step==1 or step%c['explainability']['logit_lens_every_steps']==0)
   for opt in opts:opt.zero_grad(set_to_none=True)
   budget=c['data_curriculum']['global_tokens_per_step'];seen=0;micro=0;totals=[0.]*6;bytes_total=0;layers=[];batch_hash=hashlib.sha256();all_indices=[];observed={label:0 for label in corpus.labels};layer_batches=[]
   while seen<budget:
    indices=sampler.batch(batch_size,stage['mixture_weights']);x,y,nbytes=token_budget_batch(train,indices,budget-seen,device);nll,n,aux,rz,lz,current_layers=model(x,y,telemetry,lens);count=int(n)
    loss=nll/budget+(c['embeddings_ke2']['mtp_loss_weight']*model.mtp_loss+stage['moe_aux_loss_coeff']*aux+c['moe']['router_z_coefficient']*rz+c['output']['logit_z_coefficient']*lz)*(count/budget)
    if not torch.isfinite(loss):alarm('alarm/nonfinite_loss',step);raise FloatingPointError('Nonfinite loss')
    loss.backward();totals[0]+=float(nll.detach());totals[1]+=float(aux.detach())*count;totals[2]+=float(rz.detach())*count;totals[3]+=float(lz.detach())*count;totals[4]+=float(loss.detach());totals[5]+=float(model.mtp_loss.detach())*count;seen+=count;micro+=1;bytes_total+=nbytes;batch_hash.update(y.detach().cpu().numpy().tobytes());all_indices.extend(indices[:len(y)]);layers=current_layers;layer_batches.append((count,current_layers))
    for row,index in enumerate(indices[:len(y)]):observed[corpus.slice_by_offset[train.windows[index][0]]]+=int(y[row].ne(-100).sum())
    for event in model.hooks.events:files['alarms.jsonl'].write(json.dumps(dict(event,step=step))+'\n');files['alarms.jsonl'].flush()
   layers=aggregate_layer_metrics(layer_batches,c['moe']['num_experts_per_tok'])
   for name,param in model.named_parameters():
    if param.grad is not None and not torch.isfinite(param.grad).all():alarm('alarm/nonfinite_gradient',step);raise FloatingPointError('Nonfinite gradient: '+name)
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip'],error_if_nonfinite=True,foreach=False);factor=scheduler.apply(step)
   for opt in opts:opt.step()
   model.finish_routing_update()
   for param in model.parameters():
    if not torch.isfinite(param).all():alarm('alarm/nonfinite_parameter',step);raise FloatingPointError('Nonfinite parameter')
   engine.step=step;engine.tokens_seen+=seen
   if device.type=='cuda':torch.cuda.synchronize()
   elapsed=time.perf_counter()-tick;ce=totals[0]/seen;aux=totals[1]/seen;rz=totals[2]/seen;lz=totals[3]/seen
   record={'step':step,'loss/total':totals[4],'loss/ce':ce,'loss/moe_aux':aux,'loss/router_z':rz,'loss/logit_z':lz,'loss/mtp':totals[5]/seen,'train/ce_loss':ce,'train/ce':ce,'train/total_loss':totals[4],'train/moe_aux_loss':aux,'train/router_z_loss':rz,'train/logit_z_loss':lz,'train/token_perplexity':math.exp(ce),'train/perplexity':math.exp(ce),'train/bpb':totals[0]/(math.log(2)*bytes_total),'train/valid_tokens':seen,'train/raw_utf8_bytes':bytes_total,'stability/grad_norm_preclip':float(norm),'stability/clipped':float(norm)>c['training']['gradient_clip'],'perf/step_seconds':elapsed,'perf/tokens_per_second':seen/elapsed,'layers':layers,'optimizer/adamw_lr':c['optimizer']['adamw']['learning_rate']*factor,'optimizer/muon_lr':c['optimizer']['muon']['learning_rate']*factor if c['optimizer']['optimizer_type']=='muon_adamw_hybrid' else 0.,'optimizer/wsd_factor':factor,'curriculum/stage':stage_name,'curriculum/seq_len':stage['seq_len'],'curriculum/rope_theta':stage['rope_theta'],'curriculum/moe_aux_loss_coeff':stage['moe_aux_loss_coeff'],'curriculum/router_temperature':stage['router_temperature'],'curriculum/tokens_per_step':seen,'curriculum/tokens_seen':engine.tokens_seen,'curriculum/grad_accum_steps':micro,'curriculum/transition_progress':stage['transition_progress'],'curriculum/mixture_weights':stage['mixture_weights'],'curriculum/observed_tokens_per_slice':observed,'data/window_indices':all_indices,'data/batch_sha256':batch_hash.hexdigest()}
   if step%c['evaluation']['eval_every_steps']==0 or step==c['training']['steps']:
    probe_metrics=evaluate(probe,probe_indices);v=evaluate(val,val_indices);record.update({'probe/ce':probe_metrics['ce'],'probe/bpb':probe_metrics['bpb'],'validation/ce':v['ce'],'validation/bpb':v['bpb'],'val/perplexity':v['perplexity'],'val/bpb':v['bpb'],'eval/ce':v['ce'],'eval/perplexity':v['perplexity'],'eval/bpb':v['bpb']})
    if c['data_curriculum']['weighting_mode']=='online_excess_loss':
     losses=[]
     for label in corpus.labels:
      indices=[i for i in val_indices if val.corpus.slice_by_offset[val.windows[i][0]]==label]
      if not indices:raise ValueError('Online mixing requires held-out windows for every configured slice')
      losses.append(evaluate(val,indices)['ce'])
     engine.observe_excess(losses,c['data_curriculum']['reference_losses'],step);record['curriculum/validation_slice_ce']=losses
   records.append(record);history.append(ce);files['metrics.jsonl'].write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');files['loss.txt'].write(f'{step:06d} {ce:.17g}\n');files['total_loss.txt'].write(f'{step:06d} {totals[4]:.17g}\n')
   if step%c['training']['flush_every']==0:
    for file in files.values():file.flush()
   if (c['checkpointing']['save_every_steps'] and step%c['checkpointing']['save_every_steps']==0) or (args.stop_after and step==stop):save(step)
  final=evaluate(probe,probe_indices);final_val=evaluate(val,val_indices);digest=hashlib.sha256()
  for name,tensor in sorted(model.state_dict().items()):digest.update(name.encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  complete=stop==c['training']['steps'];summary=dict(status='PASS' if complete else 'STOPPED',initial_probe=initial,final_probe=final,initial_validation=initial_val,final_validation=final_val,train_first_ce=history[0],train_last_ce=history[-1],every_train_step_decreased=all(b<a for a,b in zip(history,history[1:])),wall_seconds=time.perf_counter()-clock,nan_count=0,final_model_sha256=digest.hexdigest(),completed_steps=len(history),tokens_seen=engine.tokens_seen,note='Fixture is wiring evidence, not quality/transfer proof')
  if selected_path=='fixture' and complete and not (final['ce']<initial['ce'] and summary['every_train_step_decreased']):summary['status']='FAIL'
  (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2))
  if summary['status']=='FAIL':raise RuntimeError('Fixture loss reduction gate failed')
 except FloatingPointError as exc:
  if 'alarm/nonfinite_activation' in str(exc):alarm('alarm/nonfinite_activation',engine.step+1)
  raise
 finally:
  for file in files.values():file.close()
  corpus.close()
  if validation_corpus is not None:validation_corpus.close()
if __name__=='__main__':main()
