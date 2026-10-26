"""Comment-aware YAML checker. Open production fields are errors, never waived."""
import argparse, copy, json, math, pathlib, sys
import yaml
from source_audit import audit_text
ROOT=pathlib.Path(__file__).resolve().parents[1]

def merge(a,b):
    for k,v in b.items():
        if k not in a: raise ValueError('Unknown override field: '+k)
        if isinstance(v,dict):
            if not isinstance(a[k],dict): raise ValueError('Override type mismatch: '+k)
            merge(a[k],v)
        else:a[k]=v
    return a

def load_config(path,override=None,allow_open=False):
    text=pathlib.Path(path).read_text(encoding='utf-8-sig'); rows=audit_text(text)
    config=yaml.safe_load(text); effective={r['setting']:r for r in rows}
    expected=set(json.loads((ROOT/'harness/schema_fields.json').read_text()))
    if set(effective)!=expected:raise ValueError('Schema fields missing/unknown: '+str(sorted(set(effective)^expected)))
    if override:
        extra=pathlib.Path(override).read_text(encoding='utf-8-sig')
        for row in audit_text(extra):
            if row['setting'] not in effective:raise ValueError('Unknown override: '+row['setting'])
            effective[row['setting']]=row
        merge(config,yaml.safe_load(extra))
    # Missing source in BASE is never repaired by an overlay.
    errors=[r['setting']+': MISSING_SOURCE in base' for r in rows if 'MISSING_SOURCE' in r['issues']]
    errors += [r['setting']+': '+','.join(r['issues']) for r in effective.values() if r['issues']]
    if allow_open and errors:
        pending=set(json.loads((ROOT/'harness/pending_size_fields.json').read_text()))
        errors=[error for error in errors if not (error.split(':')[0] in pending and error.endswith('MISSING_VALUE'))]
        errors.extend(validate_new(config,allow_open=True));errors.extend(validate_completion(config,allow_open=True))
        return config,sorted(set(errors))
    if errors:return config,sorted(set(errors))
    m=config['model'];a=config['attention'];e=config['embedding'];o=config['moe'];t=config['training'];v=config['tokenizer']
    def require(ok,message):
        if not ok:errors.append(message)
    require(config['meta']['schema_version']==1 and config['meta']['profile'] in ('production_open','tiny_reference'),'unsupported schema/profile')
    def check_scalars(obj,path=''):
        for key,value in obj.items():
            name=path+'.'+key if path else key
            if isinstance(value,dict):check_scalars(value,name)
            elif isinstance(value,float):require(math.isfinite(value),name+' must be finite')
    check_scalars(config)
    for value in [m['norm_gain'],a['qk_norm'],a['bias'],o['norm_topk_prob'],o['dropless'],config['optimizer']['muon']['nesterov']]:require(type(value) is bool,'boolean setting has wrong type')
    for name,value in [('hidden_size',m['hidden_size']),('num_layers',m['num_layers']),('query_heads',a['query_heads']),('kv_heads',a['kv_heads']),('head_dim',a['qk_head_dim']),('experts',o['num_experts']),('top_k',o['num_experts_per_tok']),('sequence_length',t['sequence_length']),('steps',t['steps']),('batch_size',t['batch_size']),('routed_ffn',o['moe_intermediate_size']),('shared_ffn',o['shared_intermediate_size']),('latent_size',o['latent_size'])]:
        require(type(value) is int and value>0,name+' must be a positive integer')
    if errors:return config,errors
    require(a['qk_head_dim']>0,'QK head dimension must be positive; projected query width may differ from hidden size')
    require(a['mechanism'] in ('mha','mqa','mla') or a['query_heads']%a['kv_heads']==0,'GQA-family heads must divide by kv_heads')
    require(a['mechanism']=='mla' or int(a['qk_head_dim']*a['partial_rotary_factor'])%2==0,'Rotary QK subdimension must be even')
    require(o['num_experts_per_tok']<=o['num_experts'],'top_k exceeds expert count')
    require(type(o['shared_experts']) is int and o['shared_experts']>=0,'shared_experts must be nonnegative integer')
    require(v['pad_multiple'] in (64,128),'vocab pad_multiple must be 64 or 128')
    require(v['padded_vocab_size']==((v['vocab_size']+v['pad_multiple']-1)//v['pad_multiple'])*v['pad_multiple'],'invalid padded vocabulary')
    for key in ['bos_id','eos_id','pad_id']:require(0<=v[key]<v['vocab_size'],key+' out of vocabulary')
    require(e['output_scale']>0,'embedding output_scale must be positive')
    require(e['output_dim']==m['hidden_size'] and type(e['ke2']['output_dim']) is int and e['ke2']['output_dim']>1,'KE output shape incompatible with model width')
    require(e['ke2']['output_dim']==m['hidden_size'] or e['ke2_projection_adapter']=='linear_if_needed','KE width mismatch requires linear adapter')
    require(e['ke2']['layout']=='batch_sequence_hidden' and e['ke2']['adapter']=='none','unsupported KE layout/adapter')
    require(e['ke2']['status'] in ('pending_g4','reference_only','delivered'),'invalid KE2 status')
    require(e['type']=='kronecker' and e['implementation']=='reference_v1','only reference Kronecker v1 executable; KE2 integration requires adapter review')
    if config['meta']['profile']=='production_open':require(e['ke2']['status']=='delivered','production KE2 contract not delivered')
    require(config['precision']['compute_dtype']=='float32','only fp32 reference precision implemented; bf16/fp8 require separate validation')
    require(0<m['norm_eps']<=0.001,'norm_eps out of stable reference bounds')
    require(config['optimizer']['embedding_weight_decay']==config['optimizer']['ke2_wd'],'embedding decay alias must equal ke2_wd')
    require(config['optimizer']['optimizer_type'] in ('adamw','muon_adamw_hybrid'),'unsupported optimizer policy')
    ad=config['optimizer']['adamw'];mu=config['optimizer']['muon']
    require(0<ad['eps']<=0.001 and 0<mu['eps']<=0.001,'optimizer epsilon out of bounds')
    require(type(mu['ns_steps']) is int and mu['ns_steps']>0 and 0<=mu['momentum']<1,'invalid Muon iterations/momentum')
    require(ad['weight_decay']>=0 and mu['weight_decay']>=0,'negative weight decay')
    require(0<=ad['beta1']<1 and 0<=ad['beta2']<1,'AdamW betas invalid')
    require(ad['learning_rate']>0 and mu['learning_rate']>0,'learning rates must be positive')
    require(m['residual']=='standard','midpoint/reversible is not implemented by this reference harness')
    require(m['norm_type']=='rmsnorm' and m['norm_placement'] in ('pre_norm','peri_norm'),'unsupported normalization option')
    require(o['activation']=='swiglu' and o['scoring_func'] in ('softmax','sigmoid'),'unsupported expert/router function')
    require(m['norm_gain'],'reference requires norm gains')
    require(a['mechanism'] in ('gqa','mha','mqa','mla','sliding_window_hybrid','diff_attn') and a['dropout']==0 and a['bias'] is False,'unsupported reference attention option')
    require(a['rope_theta']>1,'rope_theta must exceed one')
    require(o['type'] in ('standard','latent'),'unsupported MoE type')
    require(type(o['dropless']) is bool and o['routed_scaling_factor']>0,'invalid dropless/output scale')
    require(o['aux_coefficient']>=0 and o['router_z_coefficient']>=0,'negative router loss coefficient')
    require(config['output']['logit_z_coefficient']>=0 and config['output']['loss_chunk_tokens']>0,'invalid output loss settings')
    require(t['gradient_clip']>0 and 0<=t['warmup_steps']<=t['steps'],'invalid clipping/warmup')
    for key in ['flush_every']:require(type(t[key]) is int and t[key]>0,key+' must be positive')
    require(type(config['explainability']['logit_lens_every_steps']) is int and config['explainability']['logit_lens_every_steps']>0,'invalid logit lens cadence')
    require(config['data']['mode'] in ('deterministic_slice','full'),'unsupported data mode')
    require(config['data']['bpb_denominator']=='exact_scored_token_bytes' and config['data']['document_attention']=='isolated' and config['data']['loss_normalization']=='valid_tokens','unsupported data accounting')
    d=config['distributed'];cur=config['data_curriculum'];xai=config['explainability']
    require(type(d['zero_stage']) is int and d['zero_stage'] in (0,3),'zero_stage must be 0 or 3 in this contract')
    require(d['moe_sharding_strategy']=='expert_data_parallel','unsupported MoE sharding strategy')
    for key in ['zero3_shard_dense_params','keep_router_in_fp32_under_zero','overlap_comm']:require(type(d[key]) is bool,'distributed.'+key+' must be boolean')
    if d['zero_stage']==3:
        require(d['keep_router_in_fp32_under_zero'] is True,'ZeRO-3 requires FP32 router pinning')
        require(d['zero3_shard_dense_params'] is True,'ZeRO-3 contract requires dense parameter sharding')
    for key in ['stage3_param_persistence_threshold','reduce_bucket_size_elements','stage3_prefetch_bucket_size_elements']:
        require(type(d[key]) is int and d[key]>=0,'distributed.'+key+' must be nonnegative integer elements')
    if d['zero_stage']==0:require(d['overlap_comm'] is False,'stage-0 deterministic reference disables communication overlap')
    require(type(cur['enabled']) is bool,'data_curriculum.enabled must be boolean')
    names=['stage_1_warmup','stage_2_main','stage_3_anneal'];stages=[cur[name] for name in names]
    fractions=[stage['step_fraction'] for stage in stages]
    require(all(type(f) in (int,float) and math.isfinite(f) and f>0 for f in fractions) and math.isclose(sum(fractions),1.,rel_tol=0,abs_tol=1e-9),'curriculum fractions must be positive and sum to 1.0')
    lengths=[stage['seq_len'] for stage in stages];thetas=[stage['rope_theta'] for stage in stages];coeffs=[stage['moe_aux_loss_coeff'] for stage in stages]
    require(all(type(n) is int and n>0 for n in lengths),'curriculum seq_len must be positive integers')
    require(lengths==sorted(lengths) and max(lengths)<=t['sequence_length'],'curriculum seq_len must be non-decreasing and <= training.sequence_length')
    require(all(theta>1 for theta in thetas) and thetas==sorted(thetas),'curriculum rope_theta must be >1 and non-decreasing')
    require(all(coeff>=0 for coeff in coeffs) and coeffs==sorted(coeffs,reverse=True),'curriculum auxiliary coefficients must be nonnegative and non-increasing')
    for key in ['enabled','track_perplexity_and_bpb','track_residual_cosine_sim','track_attention_entropy','track_moe_router_metrics','enable_logit_lens']:require(type(xai[key]) is bool,'explainability.'+key+' must be boolean')
    require(xai['track_perplexity_and_bpb'] is True,'PPL/BPB are mandatory first-class metrics in this schema')
    require(type(xai['tap_every_steps']) is int and xai['tap_every_steps']>0,'tap_every_steps must be positive')
    points=xai['sae_tap_points'];require(isinstance(points,list) and len(points)==len(set(points)) and set(points)<= {'resid_mid','mlp_out','moe_router_logits'},'unknown/duplicate SAE tap point')
    probe=xai['sae_probe'];require(type(probe['enabled']) is bool,'sae_probe.enabled must be boolean')
    require(probe['kind'] in ('random_topk_reference','trained_topk'),'unsupported native SAE probe kind')
    require(type(probe['feature_count']) is int and type(probe['top_k']) is int and 0<probe['top_k']<=probe['feature_count'],'SAE top_k must be <= feature_count')
    require(type(probe['dead_feature_window']) is int and probe['dead_feature_window']>0 and 0<=probe['dead_feature_fraction_threshold']<=1,'invalid SAE alarm window/threshold')
    if probe['enabled'] and probe['kind']=='trained_topk':require(probe['weights_path']!='none' and pathlib.Path(probe['weights_path']).is_file() and len(points)==1,'trained Top-K probe needs one tap and a valid checkpoint; use native registration for per-tap checkpoints')
    require(type(config['data']['slice_documents']) is int and config['data']['slice_documents']>=2,'slice_documents must be >=2')
    errors.extend(validate_new(config))
    errors.extend(validate_completion(config))
    return config,errors


def validate_new(c,allow_open=False):
    errors=[]
    def require(ok,message):
        if not ok:errors.append(message)
    cur=c['data_curriculum'];m=c['moe'];a=c['attention'];opt=c['optimizer'];out=c['output'];t=c['training'];emb=c['embedding']
    require(type(cur['enforce_token_batch_invariance']) is bool and cur['enforce_token_batch_invariance'],'valid-token invariance must be enabled')
    budget=cur['global_tokens_per_step']
    if not (allow_open and budget is None):require(type(budget) is int and budget>0,'global_tokens_per_step must be positive integer')
    require(cur['transition_curve'] in ('cosine','linear') and type(cur['transition_ramp_steps']) is int and cur['transition_ramp_steps']>=0,'invalid transition ramp')
    require(cur['reset_position_ids_on_eos'] is True and cur['intra_document_causal_mask'] is True,'boundary position reset and document mask are mandatory')
    require(cur['weighting_mode'] in ('static','online_excess_loss'),'unsupported weighting mode')
    require(c['precision']['compute_dtype']=='float32' and c['model']['residual']=='standard','unsupported precision/residual reference')
    require(c['meta']['schema_version']==1 and c['meta']['profile'] in ('production_open','tiny_reference'),'invalid schema/profile')
    for key in ['steps','batch_size','flush_every']:require(type(t[key]) is int and t[key]>0,'invalid training.'+key)
    require(type(t['seed']) is int and 0<=t['seed']<2**32 and 0<=t['warmup_steps']<=t['steps'] and t['gradient_clip']>0,'invalid seed/warmup/clipping')
    require(c['model']['norm_type']=='rmsnorm' and c['model']['norm_placement'] in ('pre_norm','peri_norm') and c['model']['norm_gain'] is True,'invalid normalization contract')
    require(a['mechanism'] in ('gqa','mha','mqa','mla','sliding_window_hybrid','diff_attn') and a['dropout']==0 and a['bias'] is False and a['rope_theta']>1,'invalid attention reference')
    require(m['type'] in ('standard','latent') and m['activation']=='swiglu' and m['scoring_func'] in ('softmax','sigmoid'),'invalid MoE reference')
    require(m['routed_scaling_factor']>0 and m['aux_coefficient']>=0 and m['router_z_coefficient']>=0,'invalid MoE scale/loss coefficient')
    require(out['loss_chunk_tokens']>0 and out['logit_z_coefficient']>=0,'invalid loss chunk/coefficient')
    require(opt['embedding_weight_decay']==opt['ke2_wd'],'embedding decay alias mismatch')
    for algorithm in ['adamw','muon']:require(opt[algorithm]['learning_rate']>0 and opt[algorithm]['weight_decay']>=0,'invalid optimizer rate/decay')
    require(0<=opt['adamw']['beta1']<1 and 0<=opt['adamw']['beta2']<1 and 0<=opt['muon']['momentum']<1 and opt['muon']['ns_steps']>0,'invalid optimizer moments/iterations')
    distributed=c['distributed'];require(distributed['moe_sharding_strategy']=='expert_data_parallel','invalid expert sharding contract')
    if distributed['zero_stage'] is not None:
        require(distributed['zero_stage'] in (0,3),'invalid ZeRO stage')
        if distributed['zero_stage']==3:require(distributed['keep_router_in_fp32_under_zero'] is True and distributed['zero3_shard_dense_params'] is True,'ZeRO-3 requires FP32 router pinning and dense sharding')
        if distributed['zero_stage']==0:require(distributed['overlap_comm'] is False,'deterministic stage 0 disables overlap')

    require(0<c['model']['norm_eps']<=0.001 and 0<opt['adamw']['eps']<=0.001 and 0<opt['muon']['eps']<=0.001,'invalid numerical epsilon')
    require(c['embedding']['type']=='kronecker' and opt['optimizer_type'] in ('adamw','muon_adamw_hybrid'),'unsupported embedding/optimizer')
    def finite(obj):
        if isinstance(obj,dict):return all(finite(v) for v in obj.values())
        if isinstance(obj,list):return all(finite(v) for v in obj)
        return not isinstance(obj,float) or math.isfinite(obj)
    require(finite(c),'nonfinite config value')
    labels=cur['domain_labels'];require(isinstance(labels,list) and len(labels)>0 and len(labels)==len(set(labels)) and all(isinstance(x,str) and x for x in labels),'slice labels must be distinct nonempty strings')
    require(0<=cur['min_domain_weight']<1/len(labels),'invalid minimum domain weight')
    require(cur['online_weight_lr']>0 and len(cur['reference_losses'])==len(labels),'invalid excess-loss reference dimensions/rate')
    stages=[cur[name] for name in ['stage_1_warmup','stage_2_main','stage_3_anneal']]
    for stage in stages:
        require(stage['step_fraction']>0 and stage['rope_theta']>1 and stage['moe_aux_loss_coeff']>=0,'invalid stage fractions/theta/aux')
        if stage['seq_len'] is not None:require(type(stage['seq_len']) is int and stage['seq_len']>0,'invalid stage length')
    require([stage['rope_theta'] for stage in stages]==sorted(stage['rope_theta'] for stage in stages),'stage theta must be non-decreasing')
    require([stage['moe_aux_loss_coeff'] for stage in stages]==sorted((stage['moe_aux_loss_coeff'] for stage in stages),reverse=True),'stage auxiliary coefficients must anneal')

    require(math.isclose(sum(stage['step_fraction'] for stage in stages),1.,abs_tol=1e-9),'stage fractions must sum to 1')
    for stage in stages:
        weights=stage['mixture_weights'];require(len(weights)==len(labels) and all(w>=0 for w in weights) and math.isclose(sum(weights),1.,abs_tol=1e-9),'mixture weights must sum to 1 and match slice labels')
        require(stage['router_temperature']>0,'router temperature must be positive')
    require(m['router_balance_mode'] in ('switch_aux_loss','aux_loss_free_bias','hybrid'),'unsupported router balance mode')
    require(m['bias_update_rate']>=0 and m['train_capacity_factor']>0 and m['eval_capacity_factor']>0,'invalid bias/capacity control')
    if not (allow_open and m['num_shared_experts'] is None):require(m['num_shared_experts']==m['shared_experts'],'shared expert aliases must agree')
    require(type(m['first_k_dense_replace']) is int and m['first_k_dense_replace']>=0,'dense prefix must be nonnegative integer')
    if c['model']['num_layers'] is not None:require(m['first_k_dense_replace']<=c['model']['num_layers'],'dense prefix exceeds model depth')
    require(type(m['dropless_moe']) is bool and m['dropless_moe']==m['dropless'],'dropless aliases must agree')
    require(type(a['use_qk_norm']) is bool and a['use_qk_norm']==a['qk_norm'],'QK norm aliases must agree')
    require(a['attn_logit_softcapping']>=0 and out['final_logit_softcapping']>=0,'softcaps must be nonnegative')
    require(type(emb['ke2_variance_norm']) is bool and emb['ke2_embed_scale']>0 and emb['ke2_projection_adapter'] in ('none','linear_if_needed'),'invalid KE adapter/variance guard')
    require(type(out['factorized_chunk_vocab']) is int and out['factorized_chunk_vocab']>0,'invalid factorized output chunk')
    require(type(opt['mup_enabled']) is bool and type(opt['mup_base_width']) is int and opt['mup_base_width']>0,'invalid muP width anchor')
    require(opt['mup_attention_scaling']=='inverse_head_dim' and opt['mup_output_mult']>0,'invalid muP attention/readout hooks')
    require(opt['router_lr_mult']>0 and opt['ke2_lr_mult']>0 and opt['ke2_wd']>=0,'invalid role LR/decay')
    require(c['scheduler']['type'] in ('wsd','constant') and c['scheduler']['decay_style'] in ('cosine','sqrt_1_minus_t') and 0<c['scheduler']['min_lr_ratio']<=1,'invalid WSD controls')
    if c['scheduler']['type']=='wsd':require(t['warmup_steps']<=math.ceil(cur['stage_1_warmup']['step_fraction']*t['steps']),'WSD warmup must end within stage 1')
    require(type(c['checkpointing']['save_every_steps']) is int and c['checkpointing']['save_every_steps']>=0,'invalid checkpoint cadence')
    return errors


def validate_completion(c,allow_open=False):
    errors=[]
    def require(ok,message):
        if not ok:errors.append(message)
    a=c['attention'];model=c['model'];moe=c['moe'];d=c['distributed'];e=c['evaluation'];p=c['precision'];cur=c['data_curriculum'];cp=c['checkpointing'];heads=c['embeddings_ke2']
    require(a['mechanism'] in ('gqa','mha','mqa','mla','sliding_window_hybrid','diff_attn'),'unknown attention mechanism')
    if a['mechanism']=='mha' and a['query_heads'] is not None and a['kv_heads'] is not None:require(a['query_heads']==a['kv_heads'],'MHA requires kv_heads == query_heads')
    if a['mechanism']=='mqa' and a['kv_heads'] is not None:require(a['kv_heads']==1,'MQA requires exactly one KV head')
    if a['mechanism'] in ('gqa','sliding_window_hybrid','diff_attn') and a['query_heads'] is not None and a['kv_heads'] is not None:require(a['kv_heads']>0 and a['query_heads']%a['kv_heads']==0,'query heads must be divisible by KV heads')
    require(model['norm_placement'] in ('pre_norm','peri_norm'),'unknown norm placement')
    require(model['residual_init_scale_mode'] in ('none','depth_scaled'),'unknown residual init scale')
    require(type(model['ffn_multiple_of']) is int and model['ffn_multiple_of']>0,'ffn_multiple_of must be positive')
    for value in [model['d_ff'],moe['moe_intermediate_size'],moe['shared_intermediate_size']]:
        if value is not None:require(type(value) is int and value>0 and value%model['ffn_multiple_of']==0,'dense/expert FFN widths must be positive aligned multiples')
    require(type(moe['moe_layer_freq']) is int and moe['moe_layer_freq']>0,'invalid MoE layer frequency')
    require(moe['scoring_func'] in ('softmax','sigmoid') and type(moe['norm_topk_prob']) is bool and moe['routed_scaling_factor']>0,'invalid routing score/scale')
    require(type(moe['n_group']) is int and type(moe['topk_group']) is int and 1<=moe['topk_group']<=moe['n_group'],'invalid router group budget')
    if moe['num_experts'] is not None and moe['num_experts_per_tok'] is not None:require(moe['num_experts']%moe['n_group']==0 and moe['num_experts_per_tok']<=moe['topk_group']*(moe['num_experts']//moe['n_group']),'expert groups cannot supply requested top-k')
    for key in ['qk_head_dim','v_head_dim','sliding_window_size','kv_lora_rank','q_lora_rank','qk_rope_head_dim']:
        if a[key] is not None:require(type(a[key]) is int and a[key]>0,'attention.'+key+' must be positive integer')
    require(type(a['global_attn_every_n_layers']) is int and a['global_attn_every_n_layers']>0,'invalid local/global period')
    require(type(a['num_sink_tokens']) is int and a['num_sink_tokens']>=0,'invalid sink count')
    require(type(a['kv_sharing_group_size']) is int and a['kv_sharing_group_size']>0,'invalid sharing group size')
    if model['num_layers'] is not None:require(model['num_layers']%a['kv_sharing_group_size']==0,'layer count must divide into KV-sharing groups')
    require(0<a['partial_rotary_factor']<=1,'partial_rotary_factor must lie in (0,1]')
    if a['mechanism']=='mla':
        require(all(type(a[key]) is int and a[key]>0 for key in ['kv_lora_rank','q_lora_rank','qk_rope_head_dim']) and type(a['qk_nope_head_dim']) is int and a['qk_nope_head_dim']>=0,'MLA requires positive latent/query/rotary ranks and nonnegative content width')
        if a['qk_nope_head_dim'] is not None:require(type(a['qk_nope_head_dim']) is int and a['qk_nope_head_dim']>=0,'invalid MLA nonrotary dimension')
        if all(a[key] is not None for key in ['qk_head_dim','qk_nope_head_dim','qk_rope_head_dim']):require(a['qk_head_dim']==a['qk_nope_head_dim']+a['qk_rope_head_dim'] and a['qk_rope_head_dim']%2==0,'MLA QK must equal content+even rotary dimensions')
    elif a['qk_head_dim'] is not None:
        rotary=a['qk_head_dim']*a['partial_rotary_factor'];require(rotary==int(rotary) and rotary>=2 and int(rotary)%2==0,'partial rotary width must be a positive even integer')
    scaling=a['rope_scaling'];require(scaling['type'] in ('none','linear') and scaling['factor']>=1,'only none/linear RoPE scaling implemented')
    if scaling['original_max_position_embeddings'] is not None:require(type(scaling['original_max_position_embeddings']) is int and scaling['original_max_position_embeddings']>0,'invalid original context size')
    require(heads['lm_head_mode'] in ('untied_dense','tied_dense','factorized_ke2_head'),'unknown LM head mode')
    require(type(heads['mtp_num_heads']) is int and heads['mtp_num_heads']>=0 and heads['mtp_loss_weight']>=0,'invalid MTP settings')
    for key in ['dp_size','ep_size','tp_size','cp_size']:
        if d[key] is not None:require(type(d[key]) is int and d[key]>0,'invalid 4D topology size: '+key)
    require(type(d['activation_checkpointing']) is bool,'activation_checkpointing must be boolean')
    require(cur['packing_mode']=='document_isolated','only byte-accounted document-isolated packing is implemented')
    require(cur['fim_rate']==0,'positive FIM rate needs an agreed tokenizer marker/data contract; unsupported rather than silently ignored')
    require(type(cur['stage_rng_seed_stride']) is int and cur['stage_rng_seed_stride']>=0,'invalid stage RNG stride')
    require(all(p[key]=='float32' for key in ['param_dtype','compute_dtype','grad_reduce_dtype','router_dtype','loss_dtype']) and p['allow_tf32'] is False,'reference precision requires FP32 and TF32 disabled')
    require(type(cp['save_every_steps']) is int and cp['save_every_steps']>=0 and type(cp['keep_last_n']) is int and cp['keep_last_n']>0,'invalid checkpoint cadence/retention')
    require(type(cp['save_optimizer_state']) is bool and type(cp['save_rng_and_dataloader_state']) is bool,'invalid checkpoint state flags')
    require(type(e['eval_every_steps']) is int and e['eval_every_steps']>0 and type(e['eval_batches']) is int and e['eval_batches']>0,'invalid evaluation budget/cadence')
    require(e['exclude_aux_loss_from_val'] is True and e['compute_token_ppl'] is True and e['compute_bpb'] is True,'held-out CE-only PPL/BPB are mandatory')
    return errors

def main():
    p=argparse.ArgumentParser();p.add_argument('config',nargs='?',default=ROOT/'configs/config_v0.yaml');p.add_argument('--override');p.add_argument('--resolved-out');p.add_argument('--allow-open-production-sizes',action='store_true');args=p.parse_args()
    try:c,errors=load_config(args.config,args.override,args.allow_open_production_sizes)
    except (ValueError,TypeError,KeyError,OSError,yaml.YAMLError,ZeroDivisionError) as exc:errors=['CONFIG_ERROR: '+str(exc)];c=None
    for err in errors:print(err)
    if args.allow_open_production_sizes:
        pending=[row['setting'] for row in audit_text(pathlib.Path(args.config).read_text()) if 'MISSING_VALUE' in row['issues']]
        print(f'G7 schema/source handoff: {len(pending)} pending size fields; dimension checks deferred, NOT a runnable config')
    print(f'{len(errors)} errors')
    if not errors and args.resolved_out:pathlib.Path(args.resolved_out).write_text(json.dumps(c,indent=2),encoding='utf8')
    return int(bool(errors))
if __name__=='__main__':sys.exit(main())
