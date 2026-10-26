"""Slow FP32 correctness reference. Dense expert evaluation avoids atomic dispatch."""
import array, math, pathlib, sys, contextlib, random
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from input_utils import exact_byte_buffers
sys.path.insert(0,str(pathlib.Path(__file__).parent/'vendor'))
from kronecker_embeddings import KroneckerEmbedding

class RMSNorm(nn.Module):
    def __init__(self,d,eps):super().__init__();self.weight=nn.Parameter(torch.ones(d));self.eps=eps
    def forward(self,x):return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)*self.weight

class Attention(nn.Module):
    """Reference causal attention with explicit geometry and tensor-only inference caches.

    MLA caches normalized latent KV plus raw shared rotary keys; expanded KV is temporary.
    Differential mode subtracts two softmax maps, then applies per-head RMS normalization.
    Learned sink slots are static parameters, not prefix text or per-token cache entries.
    """
    def __init__(self,c,layer=0):
        super().__init__();a=c['attention'];d=c['model']['hidden_size'];eps=c['model']['norm_eps']
        self.config=c;self.layer=layer;self.kind=a['mechanism'];self.h=a['query_heads'];self.kv=self.h if self.kind in ('mha','mla') else 1 if self.kind=='mqa' else a['kv_heads'];self.hd=a['qk_head_dim'];self.vd=a['v_head_dim'];self.theta=a['rope_theta'];self.cap=a['attn_logit_softcapping'];self.sinks=a['num_sink_tokens'];self.group=a['kv_sharing_group_size'];self.owner=layer%self.group==0
        self.local=self.kind=='sliding_window_hybrid' and (layer+1)%a['global_attn_every_n_layers']!=0
        self.rot_dim=a['qk_rope_head_dim'] if self.kind=='mla' else int(self.hd*a['partial_rotary_factor'])
        self.qnorm=RMSNorm(self.hd,eps) if a['use_qk_norm'] else nn.Identity();self.knorm=RMSNorm(self.hd,eps) if a['use_qk_norm'] and self.owner else nn.Identity()
        if self.kind=='mla':
            self.rank=a['kv_lora_rank'];self.nope=a['qk_nope_head_dim'];self.dr=a['qk_rope_head_dim'];rq=a['q_lora_rank']
            self.q_down=nn.Linear(d,rq,bias=False);self.q_low_norm=RMSNorm(rq,eps);self.q=nn.Linear(rq,self.h*self.hd,bias=False)
            if self.owner:
                self.kv_down=nn.Linear(d,self.rank+self.dr,bias=False);self.kv_low_norm=RMSNorm(self.rank,eps);self.kv_up=nn.Linear(self.rank,self.h*(self.nope+self.vd),bias=False)
        else:
            self.q=nn.Linear(d,self.h*self.hd,bias=False)
            if self.owner:self.k=nn.Linear(d,self.kv*self.hd,bias=False);self.v=nn.Linear(d,self.kv*self.vd,bias=False)
        self.out=nn.Linear(self.h*self.vd,d,bias=False)
        if self.kind=='diff_attn':
            self.q2=nn.Linear(d,self.h*self.hd,bias=False)
            if self.owner:self.k2=nn.Linear(d,self.kv*self.hd,bias=False)
            self.lambda_q1=nn.Parameter(torch.randn(self.hd)*0.1);self.lambda_k1=nn.Parameter(torch.randn(self.hd)*0.1);self.lambda_q2=nn.Parameter(torch.randn(self.hd)*0.1);self.lambda_k2=nn.Parameter(torch.randn(self.hd)*0.1);self.lambda_init=0.8-0.6*math.exp(-0.3*layer);self.diff_norm=RMSNorm(self.vd,eps)
        if self.sinks:
            self.sink_k=nn.Parameter(torch.empty(self.h*self.sinks,self.hd));self.sink_v=nn.Parameter(torch.empty(self.h*self.sinks,self.vd));nn.init.normal_(self.sink_k,std=0.02);nn.init.normal_(self.sink_v,std=0.02)
            if self.kind=='diff_attn':self.sink_k2=nn.Parameter(torch.randn(self.h*self.sinks,self.hd)*0.02)
    def rope(self,x,position_ids=None):
        r=self.rot_dim
        if not r:return x
        inv=self.theta**(-torch.arange(0,r,2,device=x.device,dtype=torch.float32)/r)
        positions=torch.arange(x.shape[-2],device=x.device)[None,:] if position_ids is None else position_ids
        scaling=self.config['attention']['rope_scaling'];positions=positions.float()/(scaling['factor'] if scaling['type']=='linear' else 1.)
        angle=positions[:,None,:,None]*inv[None,None,None,:];a,b=x[...,:r:2],x[...,1:r:2]
        rotated=torch.stack((a*angle.cos()-b*angle.sin(),a*angle.sin()+b*angle.cos()),-1).flatten(-2)
        return torch.cat((rotated,x[...,r:]),-1)
    def kv_cache_bytes_per_token(self):
        elements=self.rank+self.dr if self.kind=='mla' else self.kv*((2 if self.kind=='diff_attn' else 1)*self.hd+self.vd)
        return elements*next(self.parameters()).element_size()
    def forward(self,x,valid,telemetry,position_ids=None,document_ids=None,cache=None,use_cache=False,kv_context=None):
        if use_cache and torch.is_grad_enabled():raise ValueError('KV cache API requires no_grad inference')
        if cache is not None and not use_cache:raise ValueError('Pass use_cache=True with a KV cache')
        b,t,_=x.shape;a=self.config['attention'];past=cache['total_seen'] if cache else 0
        if cache and (cache['mechanism']!=self.kind or cache['layer']!=self.layer or cache['theta']!=self.theta):raise ValueError('Incompatible attention cache')
        if position_ids is None:position_ids=torch.arange(past,past+t,device=x.device)[None,:].expand(b,-1)
        if document_ids is None:document_ids=torch.zeros(b,t,device=x.device,dtype=torch.long)
        meta={name:torch.cat((cache[name],value),-1) if cache else value for name,value in [('positions',position_ids),('documents',document_ids),('valid',valid)]}
        q_input=self.q_low_norm(self.q_down(x)) if self.kind=='mla' else x
        q=self.qnorm(self.q(q_input).view(b,t,self.h,self.hd).transpose(1,2))
        q=torch.cat((q[...,:self.nope],self.rope(q[...,self.nope:],position_ids)),-1) if self.kind=='mla' else self.rope(q,position_ids)
        raw={};group_id=self.layer//self.group
        if not self.owner:
            if kv_context is None or group_id not in kv_context:raise ValueError('KV-sharing follower requires its owner pack in this forward')
            pack=kv_context[group_id];k=pack['k'];v=pack['v'];k2=pack.get('k2');meta=pack['meta'];raw=pack['raw']
        elif self.kind=='mla':
            compressed=self.kv_down(x);latent=self.kv_low_norm(compressed[...,:self.rank]);rotary=compressed[...,self.rank:]
            if cache:latent=torch.cat((cache['latent'],latent),1);rotary=torch.cat((cache['rope_key'],rotary),1)
            raw={'latent':latent,'rope_key':rotary};length=latent.shape[1];up=self.kv_up(latent).view(b,length,self.h,self.nope+self.vd).transpose(1,2);content,v=up[...,:self.nope],up[...,self.nope:]
            k=self.knorm(torch.cat((content,rotary[:,None].expand(-1,self.h,-1,-1)),-1));k=torch.cat((k[...,:self.nope],self.rope(k[...,self.nope:],meta['positions'])),-1);k2=None
        else:
            k=self.rope(self.knorm(self.k(x).view(b,t,self.kv,self.hd).transpose(1,2)),position_ids);v=self.v(x).view(b,t,self.kv,self.vd).transpose(1,2);k2=None
            if self.kind=='diff_attn':k2=self.rope(self.knorm(self.k2(x).view(b,t,self.kv,self.hd).transpose(1,2)),position_ids)
            if cache:
                k=torch.cat((cache['k'],k),2);v=torch.cat((cache['v'],v),2)
                if k2 is not None:k2=torch.cat((cache['k2'],k2),2)
            raw={'k':k,'v':v}
            if k2 is not None:raw['k2']=k2
        if self.owner and kv_context is not None:kv_context[group_id]={'k':k,'v':v,'k2':k2,'meta':meta,'raw':raw}
        repeats=self.h//k.shape[1];k=k.repeat_interleave(repeats,1);v=v.repeat_interleave(repeats,1)
        if k2 is not None:k2=k2.repeat_interleave(repeats,1)
        key_indices=torch.arange(past+t-meta['valid'].shape[-1],past+t,device=x.device);query_indices=torch.arange(past,past+t,device=x.device)
        causal=query_indices[:,None]>=key_indices[None,:];mask=causal[None,None]&meta['valid'][:,None,None,:]&document_ids[:,None,:,None].eq(meta['documents'][:,None,None,:])
        if self.local:mask=mask & ((query_indices[:,None]-key_indices[None,:])<a['sliding_window_size'])[None,None]
        fallback=query_indices[:,None].eq(key_indices[None,:]);mask=mask | ((~valid)[:,None,:,None]&fallback[None,None])
        if self.sinks:
            k=torch.cat((self.sink_k.view(self.h,self.sinks,self.hd)[None].expand(b,-1,-1,-1),k),2);v=torch.cat((self.sink_v.view(self.h,self.sinks,self.vd)[None].expand(b,-1,-1,-1),v),2)
            if k2 is not None:k2=torch.cat((self.sink_k2.view(self.h,self.sinks,self.hd)[None].expand(b,-1,-1,-1),k2),2)
            mask=torch.cat((torch.ones(b,1,t,self.sinks,device=x.device,dtype=torch.bool),mask),-1)
        denominator=self.hd if self.config['optimizer']['mup_enabled'] else math.sqrt(self.hd)
        def probabilities(query,key):
            logits=query.float()@key.float().transpose(-1,-2)/denominator
            if self.cap>0:logits=self.cap*torch.tanh(logits/self.cap)
            return logits.masked_fill(~mask,float('-inf')).softmax(-1)
        p=probabilities(q,k);weights=p
        if self.kind=='diff_attn':
            q2=self.rope(self.qnorm(self.q2(x).view(b,t,self.h,self.hd).transpose(1,2)),position_ids);p2=probabilities(q2,k2)
            lam=(self.lambda_q1*self.lambda_k1).sum().exp()-(self.lambda_q2*self.lambda_k2).sum().exp()+self.lambda_init;weights=p-lam*p2
        y=weights@v
        if self.kind=='diff_attn':y=self.diff_norm(y)*(1-self.lambda_init)
        stats={}
        if telemetry:
            stats['attn/qk_norm_scale']=float(q.detach().square().mean().sqrt());stats['attn/kv_cache_bytes_per_token']=self.kv_cache_bytes_per_token();stats['attn/is_local']=self.local;stats['attn/effective_kv_heads']=self.kv;stats['attn/context_extension_ratio']=float((position_ids.max()+1)/a['rope_scaling']['original_max_position_embeddings'])
            ent=-(p*p.clamp_min(1e-30).log()).sum(-1)
            if self.kind=='diff_attn':ent=(ent-(p2*p2.clamp_min(1e-30).log()).sum(-1))/2
            stats['attention/entropy_per_head']=((ent*valid[:,None,:]).sum((0,2))/valid.sum()).detach().tolist()
            if self.sinks:stats['attn/sink_mass']=float((p[...,:self.sinks].sum(-1)*valid[:,None,:]).sum()/(valid.sum()*self.h))
        output=self.out(y.transpose(1,2).reshape(b,t,self.h*self.vd))
        if not use_cache:return output,stats
        if self.local:
            limit=a['sliding_window_size']
            raw={key:value[:,-limit:] if self.kind=='mla' else value[:,:,-limit:] for key,value in raw.items()};meta={key:value[:,-limit:] for key,value in meta.items()}
        result=dict(raw,**meta,total_seen=past+t,mechanism=self.kind,layer=self.layer,theta=self.theta)
        return output,stats,result

class Expert(nn.Module):
    def __init__(self,d,f,activation_checkpointing=False):super().__init__();self.activation_checkpointing=activation_checkpointing;self.gate=nn.Linear(d,f,bias=False);self.up=nn.Linear(d,f,bias=False);self.down=nn.Linear(f,d,bias=False)
    def forward(self,x):
        def compute(value):return self.down(F.silu(self.gate(value))*self.up(value))
        return checkpoint(compute,x,use_reentrant=False) if self.activation_checkpointing and torch.is_grad_enabled() else compute(x)

class MoE(nn.Module):
    def __init__(self,c):
        super().__init__();m=c['moe'];d=c['model']['hidden_size'];z=m['latent_size'] if m['type']=='latent' else d
        self.k=m['num_experts_per_tok'];self.renorm=m['norm_topk_prob'];self.scale=m['routed_scaling_factor'];self.settings=m;self.temperature=1.;self.router=nn.Linear(d,m['num_experts'],bias=False)
        self.register_buffer('router_bias',torch.zeros(m['num_experts']));self.register_buffer('pending_counts',torch.zeros(m['num_experts']))
        self.compress=nn.Linear(d,z,bias=False) if z!=d else nn.Identity();self.expand=nn.Linear(z,d,bias=False) if z!=d else nn.Identity()
        self.experts=nn.ModuleList([Expert(z,m['moe_intermediate_size'],c['distributed']['activation_checkpointing']) for _ in range(m['num_experts'])]);self.shared=nn.ModuleList([Expert(d,m['shared_intermediate_size'],c['distributed']['activation_checkpointing']) for _ in range(m['shared_experts'])])
    def forward(self,x,valid,hooks=None,layer=0):
        logits=self.router(x).float()
        if hooks is not None:hooks.capture('moe_router_logits',layer,logits,valid)
        p=(logits/self.temperature).softmax(-1) if self.settings['scoring_func']=='softmax' else (logits/self.temperature).sigmoid();selection=p+self.router_bias.detach() if self.settings['router_balance_mode'] in ('aux_loss_free_bias','hybrid') else p;n_group=self.settings['n_group'];group_width=selection.shape[-1]//n_group
        if n_group>1:
            scores=selection.reshape(*selection.shape[:-1],n_group,group_width).amax(-1);groups=scores.argsort(dim=-1,descending=True,stable=True)[...,:self.settings['topk_group']]
            allowed=groups[...,None].eq(torch.arange(n_group,device=x.device)).any(-2).repeat_interleave(group_width,-1);selection=selection.masked_fill(~allowed,float('-inf'))
        selected=selection.argsort(dim=-1,descending=True,stable=True)[...,:self.k]
        # Comparison/reduction constructs a mask without scatter/index_add or tied topk ambiguity.
        chosen=selected[...,None].eq(torch.arange(len(self.experts),device=x.device)).any(-2)&valid[...,None]
        requested=chosen.sum((0,1)).float()
        if self.training:self.pending_counts.add_(requested.detach())
        requests=chosen.clone()
        if not self.settings['dropless_moe']:
            capacity=math.ceil((self.settings['train_capacity_factor'] if self.training else self.settings['eval_capacity_factor'])*int(valid.sum())*self.k/len(self.experts))
            flat=chosen.flatten(0,1);chosen=chosen & (flat.cumsum(0).reshape_as(chosen)<=capacity)
        weights=p*chosen
        if self.renorm:weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-30)
        z=self.compress(x);out=torch.zeros_like(z)
        for i,expert in enumerate(self.experts):out=out+expert(z)*weights[...,i,None]
        out=self.expand(out)*self.scale
        routed_norm=out.detach().norm();shared_out=torch.zeros_like(out)
        for expert in self.shared:shared_out=shared_out+expert(x)*valid[...,None]
        out=out+shared_out
        n=valid.sum();counts=chosen.sum((0,1)).float();fraction=counts/(n*self.k);mean=(p*valid[...,None]).sum((0,1))/n
        aux=len(self.experts)*((requested/(n*self.k))*mean).sum()
        if self.settings['router_balance_mode']=='aux_loss_free_bias':aux=aux*0
        router_z=(logits.logsumexp(-1).square()*valid).sum()/n
        entropy=-(fraction*fraction.clamp_min(1e-30).log()).sum()/math.log(len(self.experts)) if len(self.experts)>1 else fraction.sum()*0
        stats={'moe/expert_counts':counts.detach().tolist(),'moe/utilization_entropy':float(entropy.detach()),'moe/max_load_violation':float((counts.max()/(n*self.k/len(self.experts))-1).detach()),'moe/dropped_token_ratio':float((valid & ~chosen.any(-1)).sum()/n),'moe/dropped_slot_ratio':float((requests.sum()-chosen.sum())/(n*self.k)),'moe/shared_expert_norm_ratio':float(shared_out.detach().norm()/(routed_norm+1e-12)),'moe/router_bias_max_abs':float(self.router_bias.abs().max()),'moe/dead_expert_count':int(counts.eq(0).sum()),'moe/router_z_loss':float(router_z.detach())}
        return out,aux,router_z,stats
    @torch.no_grad()
    def finish_update(self):
        if self.settings['router_balance_mode'] in ('aux_loss_free_bias','hybrid') and self.pending_counts.sum()>0:self.router_bias.add_(self.settings['bias_update_rate']*torch.sign(self.pending_counts.mean()-self.pending_counts))
        self.pending_counts.zero_()

class Block(nn.Module):
    def __init__(self,c,layer=0):super().__init__();d=c['model']['hidden_size'];eps=c['model']['norm_eps'];self.n1=RMSNorm(d,eps);self.n2=RMSNorm(d,eps);self.attn=Attention(c,layer);self.post_attn=RMSNorm(d,eps) if c['model']['norm_placement']=='peri_norm' else nn.Identity();self.post_mlp=RMSNorm(d,eps) if c['model']['norm_placement']=='peri_norm' else nn.Identity();self.moe=DensePrefix(c) if layer<c['moe']['first_k_dense_replace'] or (layer-c['moe']['first_k_dense_replace'])%c['moe']['moe_layer_freq'] else MoE(c)
    def forward(self,x,valid,telemetry,hooks=None,layer=0,position_ids=None,document_ids=None,kv_context=None,cache=None,use_cache=False):
        before=x;result=self.attn(self.n1(x),valid,telemetry,position_ids,document_ids,cache=cache,use_cache=use_cache,kv_context=kv_context)
        if use_cache:xatt,stats,present=result
        else:xatt,stats=result
        x=self.post_attn(x+xatt)
        if hooks is not None:hooks.capture('resid_mid',layer,x,valid)
        out,aux,z,routing=self.moe(self.n2(x),valid,hooks,layer)
        if hooks is not None:hooks.capture('mlp_out',layer,out,valid)
        x=self.post_mlp(x+out);stats.update(routing)
        if not torch.isfinite(x).all():raise FloatingPointError('alarm/nonfinite_activation')
        if telemetry:
            stats['residual/l2_mean']=float(x.norm(dim=-1)[valid].mean().detach());stats['residual/input_output_cosine']=float(F.cosine_similarity(before,x,dim=-1)[valid].mean().detach())
        return (x,aux,z,stats,present) if use_cache else (x,aux,z,stats)

class Model(nn.Module):
    def __init__(self,c,tokenizer,embedding_provider=None):
        super().__init__();self.c=c;self.hooks=ActivationHookManager(c['explainability']);d=c['model']['hidden_size'];v=c['tokenizer']['vocab_size'];pos=c['embedding']['pos_dim'];bb,ll=exact_byte_buffers(tokenizer,pos);lengths=array.array('h',ll)
        if sys.byteorder!='little':lengths.byteswap()
        base=embedding_provider or KroneckerEmbedding(v,c['embedding']['ke2']['output_dim'],pos_dim=pos,mode='dynamic',byte_buffer=torch.frombuffer(bb,dtype=torch.uint8).clone().reshape(v,pos),length_buffer=torch.frombuffer(lengths,dtype=torch.int16).clone())
        self.embedding=KE2Adapter(base,c['embedding']['ke2']['output_dim'],d,c)
        self.blocks=nn.ModuleList([Block(c,i) for i in range(c['model']['num_layers'])]);self.norm=RMSNorm(d,c['model']['norm_eps']);self.head=nn.Linear(d,c['tokenizer']['padded_vocab_size'],bias=False) if c['embeddings_ke2']['lm_head_mode']=='untied_dense' else None;self.mtp_heads=nn.ModuleList([nn.Linear(d,d,bias=False) for _ in range(c['embeddings_ke2']['mtp_num_heads'])]);self.mtp_loss=None
        for name,param in self.named_parameters():
            if param.ndim==2 and not name.startswith('embedding.'):
                scale=math.sqrt(c['optimizer']['mup_base_width']/param.shape[1]) if c['optimizer']['mup_enabled'] else 1.
                nn.init.normal_(param,std=0.02*scale)
                if c['model']['residual_init_scale_mode']=='depth_scaled' and (name.endswith('attn.out.weight') or name.endswith('.down.weight')):param.data.mul_(1/math.sqrt(2*c['model']['num_layers']))
    @torch.no_grad()
    def finish_routing_update(self):
        for module in self.modules():
            if isinstance(module,MoE):module.finish_update()
    def project_logits(self,x):
        if self.head is not None:logits=self.head(x).float()
        else:
            v=self.c['tokenizer']['vocab_size'];mode=self.c['embeddings_ke2']['lm_head_mode'];chunk=v if mode=='tied_dense' else self.c['output']['factorized_chunk_vocab'];pieces=[]
            for start in range(0,v,chunk):
                ids=torch.arange(start,min(v,start+chunk),device=x.device)
                def project(h,token_ids):return F.linear(h,self.embedding(token_ids)*self.c['embedding']['output_scale']).float()
                pieces.append(checkpoint(project,x,ids,use_reentrant=False) if torch.is_grad_enabled() else project(x,ids))
            logits=torch.cat(pieces,-1)
        opt=self.c['optimizer']
        if opt['mup_enabled']:logits=logits*opt['mup_output_mult']/(self.c['model']['hidden_size']/opt['mup_base_width'])
        cap=self.c['output']['final_logit_softcapping']
        return cap*torch.tanh(logits/cap) if cap>0 else logits
    def losses(self,x,y):
        nll=x.new_zeros(());z=x.new_zeros(());chunk=self.c['output']['loss_chunk_tokens'];v=self.c['tokenizer']['vocab_size']
        for i in range(0,len(y),chunk):
            def part(h,target):
                logits=self.project_logits(h)[:,:v]
                return torch.stack((F.cross_entropy(logits,target,ignore_index=-100,reduction='sum'),(logits.logsumexp(-1).square()*target.ne(-100)).sum()))
            vals=checkpoint(part,x[i:i+chunk],y[i:i+chunk],use_reentrant=False) if torch.is_grad_enabled() else part(x[i:i+chunk],y[i:i+chunk]);nll=nll+vals[0];z=z+vals[1]
        return nll,z
    def forward(self,ids,labels=None,telemetry=False,lens=False,position_ids=None,document_ids=None,past_key_values=None,use_cache=False):
        if use_cache and torch.is_grad_enabled():raise ValueError('Model KV cache API requires torch.no_grad()')
        if use_cache and self.training:raise ValueError('Model KV cache API requires model.eval()')
        if past_key_values is not None and len(past_key_values)!=len(self.blocks):raise ValueError('past_key_values must contain one entry per layer')
        if past_key_values is not None and (position_ids is None or document_ids is None):raise ValueError('Cached model decode requires explicit position_ids and document_ids; reset both at document boundaries')
        valid=ids.ne(self.c['tokenizer']['pad_id']) if labels is None else labels.ne(-100)&ids.ne(self.c['tokenizer']['pad_id'])
        labels=torch.full_like(ids,-100) if labels is None else labels.masked_fill(~valid,-100)
        if position_ids is None or document_ids is None:
            positions,documents=boundary_metadata(ids,self.c['tokenizer']['eos_id'],self.c['tokenizer']['bos_id']);position_ids=positions if position_ids is None else position_ids;document_ids=documents if document_ids is None else document_ids
        self.hooks.begin(telemetry);x=self.embedding(ids)*self.c['embedding']['output_scale'];aux=[];zs=[];layers=[];lens_states=[];kv_context={};present=[]
        for layer,block in enumerate(self.blocks):
            result=block(x,valid,telemetry,self.hooks,layer,position_ids,document_ids,kv_context,cache=None if past_key_values is None else past_key_values[layer],use_cache=use_cache)
            if use_cache:x,a,z,stats,cache_out=result;present.append(cache_out)
            else:x,a,z,stats=result
            if isinstance(block.moe,MoE):aux.append(a);zs.append(z)
            if lens:
                lens_states.append(x.detach())
                with torch.no_grad():readout,_=self.losses(self.norm(x).flatten(0,1),labels.flatten());stats['logit_lens/ce']=float(readout/valid.sum())
            layers.append(stats)
        if not torch.isfinite(x).all():raise FloatingPointError('alarm/nonfinite_activation')
        if use_cache:return self.project_logits(self.norm(x)[:,-1:]),tuple(present)
        if lens:
            with torch.no_grad():
                final=self.norm(x).detach().flatten(0,1);flat_valid=valid.flatten();chunk=self.c['output']['loss_chunk_tokens'];vocab=self.c['tokenizer']['vocab_size']
                for layer,state in enumerate(lens_states):
                    intermediate=self.norm(state).flatten(0,1);kl=final.new_zeros(())
                    for start in range(0,len(final),chunk):
                        target=self.project_logits(final[start:start+chunk])[:,:vocab].log_softmax(-1)
                        current=self.project_logits(intermediate[start:start+chunk])[:,:vocab].log_softmax(-1)
                        kl=kl+((target.exp()*(target-current)).sum(-1)*flat_valid[start:start+chunk]).sum()
                    layers[layer]['explain/logit_lens_kl']=float(kl/valid.sum())
        for layer,stats in enumerate(layers):
            if not self.c['explainability']['track_residual_cosine_sim']:stats.pop('residual/input_output_cosine',None)
            if not self.c['explainability']['track_attention_entropy']:stats.pop('attention/entropy_per_head',None)
            if not self.c['explainability']['track_moe_router_metrics']:
                for key in list(stats):
                    if key.startswith('moe/'):stats.pop(key)
            aliases={'residual/input_output_cosine':'explain/residual_cosine','residual/l2_mean':'explain/residual_l2','attention/entropy_per_head':'explain/attention_entropy_per_head','moe/utilization_entropy':'explain/expert_utilization_entropy','moe/max_load_violation':'explain/max_load_violation','moe/dead_expert_count':'explain/dead_expert_count','moe/router_z_loss':'explain/router_z','logit_lens/ce':'explain/logit_lens_ce'}
            for source,target in aliases.items():
                if source in stats:stats[target]=stats[source]
            stats['explain/sae']=[dict(tap_point=point,**{'explain/sae_'+key:value for key,value in metrics.items()}) for (point,index),metrics in self.hooks.metrics.items() if index==layer]
            if not self.c['explainability']['enabled']:
                for key in list(stats):
                    if key.startswith('explain/'):stats.pop(key)
        self.mtp_loss=x.sum()*0
        for horizon,transform in enumerate(self.mtp_heads,1):
            targets=torch.full_like(labels,-100)
            if horizon<labels.shape[1]:
                shifted=labels[:,horizon:];same=document_ids[:,:-horizon].eq(document_ids[:,horizon:]);targets[:,:-horizon]=shifted.masked_fill(~same,-100)
            targets=targets.masked_fill(~valid,-100)
            auxiliary,_=self.losses(self.norm(transform(x)).flatten(0,1),targets.flatten());self.mtp_loss=self.mtp_loss+auxiliary/targets.ne(-100).sum().clamp_min(1)
        if len(self.mtp_heads):self.mtp_loss=self.mtp_loss/len(self.mtp_heads)
        nll,z=self.losses(self.norm(x).flatten(0,1),labels.flatten());n=valid.sum()
        return nll,n,torch.stack(aux).mean() if aux else nll*0,torch.stack(zs).mean() if zs else nll*0,z/n,layers


class TopKSAEProbe:
    """Inference-only probe; random mode is NOT a trained SAE or interpretable feature claim."""
    def __init__(self,width,features,k,kind='random_topk_reference',weights_path='none'):
        self.k=k;self.kind=kind
        if kind=='trained_topk':
            state=torch.load(weights_path,map_location='cpu',weights_only=True)
            self.encoder=state['encoder'].detach().float().clone();self.decoder=state['decoder'].detach().float().clone();self.bias=state.get('bias',torch.zeros(features)).detach().float().clone()
            if self.encoder.shape!=(width,features) or self.decoder.shape!=(features,width) or self.bias.shape!=(features,):raise ValueError('SAE checkpoint dimensions incompatible with tap')
        else:
            generator=torch.Generator(device='cpu').manual_seed(1729+width)
            self.encoder=torch.randn(width,features,generator=generator)/math.sqrt(width);self.decoder=self.encoder.T.clone();self.bias=torch.zeros(features)
        if not all(torch.isfinite(x).all() for x in [self.encoder,self.decoder,self.bias]):raise ValueError('Nonfinite SAE checkpoint')
    @torch.no_grad()
    def __call__(self,activation):
        x=activation.reshape(-1,activation.shape[-1]).float();encoder=self.encoder.to(x.device);decoder=self.decoder.to(x.device);bias=self.bias.to(x.device)
        z=F.relu(x@encoder+bias);selected=z.argsort(dim=-1,descending=True,stable=True)[:,:self.k]
        keep=selected[...,None].eq(torch.arange(z.shape[-1],device=x.device)).any(-2);sparse=z*keep;reconstruction=sparse@decoder
        return {'reconstruction_mse':float((reconstruction-x).square().mean()),'feature_l0':float(sparse.ne(0).float().sum(-1).mean())},sparse.ne(0).any(0).cpu()

class ActivationHookManager:
    """Bounded native SAE-ready hooks. Callback tensors own storage and never carry gradients."""
    def __init__(self,config):
        self.config=config;self.enabled=config['enabled'];self.points=set(config['sae_tap_points']);self.callbacks=[];self.probes={};self.histories={};self.active=False;self.snapshots={};self.metrics={};self.events=[]
    def register(self,callback):
        self.callbacks.append(callback)
        return lambda:self.callbacks.remove(callback)
    def register_sae_probe(self,point,layer,probe):
        if point not in self.points:raise ValueError('Tap is not enabled: '+point)
        self.probes[(point,layer)]=probe
    def begin(self,active):self.active=bool(active and self.enabled);self.snapshots={};self.metrics={};self.events=[]
    @torch.no_grad()
    def capture(self,point,layer,tensor,valid):
        if not self.active or point not in self.points:return
        detached=tensor.detach()[valid].clone();key=(point,layer);self.snapshots[key]=detached
        for callback in self.callbacks:
            with isolated_probe_rng(detached.device):callback(point,layer,detached.clone())
        p=self.config['sae_probe']
        if key not in self.probes and p['enabled']:self.probes[key]=TopKSAEProbe(detached.shape[-1],p['feature_count'],p['top_k'],p['kind'],p['weights_path'])
        if key in self.probes:
            with isolated_probe_rng(detached.device):stats,active=self.probes[key](detached)
            history=self.histories.setdefault(key,[]);history.append(active)
            if len(history)>p['dead_feature_window']:history.pop(0)
            observed=torch.stack(history).any(0);stats['dead_feature_count']=int((~observed).sum());stats['dead_feature_fraction']=float((~observed).float().mean());stats['observation_count']=len(history)
            self.metrics[key]=stats
            if len(history)>=p['dead_feature_window'] and stats['dead_feature_fraction']>=p['dead_feature_fraction_threshold']:self.events.append({'name':'alarm/sae_dead_features','point':point,'layer':layer})


def boundary_metadata(ids,eos,bos):
    starts=ids.eq(bos);starts[:,0]=True
    starts[:,1:]=starts[:,1:] | ids[:,:-1].eq(eos)
    documents=starts.long().cumsum(-1);absolute=torch.arange(ids.shape[1],device=ids.device).expand_as(ids)
    latest=torch.where(starts,absolute,torch.zeros_like(absolute)).cummax(-1).values
    return absolute-latest,documents

class KE2Adapter(nn.Module):
    def __init__(self,base,input_dim,output_dim,c):
        super().__init__();self.base=base;self.settings=c['embedding'];self.eps=c['model']['norm_eps'];self.input_dim=input_dim
        if input_dim!=output_dim and self.settings['ke2_projection_adapter']!='linear_if_needed':raise ValueError('KE2 width mismatch needs projection adapter')
        self.projection=nn.Linear(input_dim,output_dim,bias=False) if input_dim!=output_dim else nn.Identity()
    def forward(self,ids):
        x=self.base(ids).float()
        if x.shape!=(*ids.shape,self.input_dim):raise ValueError('G4 provider returned an incompatible lookup shape')
        if self.settings['ke2_variance_norm']:
            centered=x-x.mean(-1,keepdim=True);x=centered*torch.rsqrt(centered.square().mean(-1,keepdim=True)+self.eps)
        return self.projection(x)*self.settings['ke2_embed_scale']

class DensePrefix(nn.Module):
    def __init__(self,c):super().__init__();self.expert=Expert(c['model']['hidden_size'],c['model']['d_ff'],c['distributed']['activation_checkpointing'])
    def forward(self,x,valid,hooks=None,layer=0):
        output=self.expert(x)*valid[...,None];zero=output.sum()*0
        return output,zero,zero,{'moe/dense_prefix':True,'moe/shared_expert_norm_ratio':0.,'moe/router_bias_max_abs':0.}

@contextlib.contextmanager
def isolated_probe_rng(device):
    import numpy as np
    python_state=random.getstate();numpy_state=np.random.get_state();devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):yield
    finally:random.setstate(python_state);np.random.set_state(numpy_state)
