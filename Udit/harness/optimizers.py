"""Own FP32 Muon reference plus explicit AdamW groups for torch 2.5.1.
Algorithm source: PyTorch 2.9 Muon docs, original rectangular scaling.
No BF16 orthogonalization, fused kernels or optimizer-state sharding claimed.
"""
import math
import torch
from model import RMSNorm

class ReferenceMuon(torch.optim.Optimizer):
 def __init__(self,params,c):
  super().__init__(params,dict(lr=c['learning_rate'],peak_lr=c['learning_rate'],weight_decay=c['weight_decay'],momentum=c['momentum'],nesterov=c['nesterov'],ns_steps=c['ns_steps'],eps=c['eps']))
 @torch.no_grad()
 def step(self,closure=None):
  if closure is not None:raise ValueError('Closures not supported by reference Muon')
  for group in self.param_groups:
   for p in group['params']:
    if p.ndim!=2:raise ValueError('Muon received non-matrix')
    if p.grad is None:raise ValueError('Missing gradient for assigned Muon parameter')
    g=p.grad.float();state=self.state[p]
    if 'momentum' not in state:state['momentum']=torch.zeros_like(g)
    buf=state['momentum'];buf.mul_(group['momentum']).add_(g)
    z=g+group['momentum']*buf if group['nesterov'] else buf.clone()
    transposed=z.shape[0]>z.shape[1]
    if transposed:z=z.T
    z=z/(z.norm()+group['eps'])
    for _ in range(group['ns_steps']):
     a=z@z.T;z=3.4445*z+(-4.775*a+2.0315*(a@a))@z
    if transposed:z=z.T
    z=z*math.sqrt(max(1,p.shape[0]/p.shape[1]))
    if not torch.isfinite(z).all():raise RuntimeError('Nonfinite Muon update')
    p.mul_(1-group['lr']*group['weight_decay']).add_(z,alpha=-group['lr'])


def build_optimizers(model,c):
    norm={id(p) for module in model.modules() if isinstance(module,RMSNorm) for p in module.parameters(recurse=False)}
    roles={name:[] for name in ['dense_and_expert_2d','norms_and_biases_1d','moe_router_gate','ke2_kronecker_factors']}
    names={'muon':[],'adamw_decay':[],'adamw_no_decay':[]};muon_groups=[];adam_groups=[]
    width=model.c['model']['hidden_size']/c['mup_base_width'] if c['mup_enabled'] else 1.
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('embedding.'):role='ke2_kronecker_factors'
        elif '.router.' in name:role='moe_router_gate'
        elif id(p) in norm or p.ndim<2 or name.endswith('.bias'):role='norms_and_biases_1d'
        else:role='dense_and_expert_2d'
        roles[role].append(name)
        muon=role=='dense_and_expert_2d' and not name.startswith('head.') and 'sink_' not in name and c['optimizer_type']=='muon_adamw_hybrid'
        settings=c['muon'] if muon else c['adamw'];mult=1.
        if role=='dense_and_expert_2d' and not name.startswith('head.'):mult/=width
        if role=='moe_router_gate':mult=c['router_lr_mult'];assert p.dtype==torch.float32
        if role=='ke2_kronecker_factors':mult=c['ke2_lr_mult']
        wd=c['ke2_wd'] if role=='ke2_kronecker_factors' else 0. if role in ('norms_and_biases_1d','moe_router_gate') else settings['weight_decay']
        group=dict(params=[p],lr=settings['learning_rate']*mult,peak_lr=settings['learning_rate']*mult,weight_decay=wd,role=role,parameter_name=name,lr_multiplier=mult)
        if muon:muon_groups.append(group);names['muon'].append(name)
        else:adam_groups.append(group);names['adamw_no_decay' if wd==0 else 'adamw_decay'].append(name)
    assigned=[name for group in roles.values() for name in group]
    if len(assigned)!=len(set(assigned)) or set(assigned)!={n for n,p in model.named_parameters() if p.requires_grad}:raise ValueError('Optimizer role coverage/overlap error')
    opts=[];order=[]
    if muon_groups:opts.append(ReferenceMuon(muon_groups,c['muon']));order.append('muon')
    if adam_groups:opts.append(torch.optim.AdamW(adam_groups,betas=(c['adamw']['beta1'],c['adamw']['beta2']),eps=c['adamw']['eps'],foreach=False));order.append('adamw')
    names['optimizer_order']=order;names['roles']=roles
    return opts,names

class WSDScheduler:
    def __init__(self,optimizers,config):self.optimizers=optimizers;self.config=config;self.step=0;self.factor=0.
    def at(self,step):
        c=self.config;t=c['training'];warm=t['warmup_steps']
        if c['scheduler']['type']=='constant':return 1.
        if step<=warm:return step/max(1,warm)
        start=math.ceil((c['data_curriculum']['stage_1_warmup']['step_fraction']+c['data_curriculum']['stage_2_main']['step_fraction'])*t['steps'])
        if step<=start:return 1.
        progress=min(1.,(step-start)/max(1,t['steps']-start))
        decay=(1+math.cos(math.pi*progress))/2 if c['scheduler']['decay_style']=='cosine' else math.sqrt(1-progress)
        return c['scheduler']['min_lr_ratio']+(1-c['scheduler']['min_lr_ratio'])*decay
    def apply(self,step):
        self.step=step;self.factor=self.at(step)
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:group['lr']=group['peak_lr']*self.factor
        return self.factor
    def state_dict(self):return dict(step=self.step,factor=self.factor)
    def load_state_dict(self,state):self.step=state['step'];self.factor=state['factor']
