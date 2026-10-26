import hashlib,json,pathlib
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def exact_byte_buffers(path,pos_dim=32):
    """Invert ByteLevel alphabet; never decode partial UTF-8 through replacement text."""
    artifact=json.loads(pathlib.Path(path).read_text(encoding='utf-8-sig'))
    if artifact['decoder']['type']!='ByteLevel':raise ValueError('Expected pinned ByteLevel tokenizer')
    vocab=artifact['model']['vocab'];size=max(vocab.values())+1
    if size!=131072:raise ValueError('Unexpected vocabulary size')
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256));cs=bs[:];n=0
    for b in range(256):
        if b not in bs:bs.append(b);cs.append(256+n);n+=1
    inverse={chr(c):b for b,c in zip(bs,cs)}
    specials={x['id']:x['content'] for x in artifact['added_tokens'] if x['special']}
    byte_buffer=bytearray(size*pos_dim);lengths=[]
    by_id={i:piece for piece,i in vocab.items()}
    if set(by_id)!=set(range(size)):raise ValueError('Noncontiguous token IDs')
    for i in range(size):
        raw=specials[i].encode('utf-8') if i in specials else bytes(inverse[ch] for ch in by_id[i])
        if not 0<len(raw)<=pos_dim:raise ValueError(f'Token {i} has invalid byte length: {len(raw)}')
        lengths.append(len(raw));byte_buffer[i*pos_dim:i*pos_dim+len(raw)]=raw
    return byte_buffer,lengths


# Original dataset gzip hashes mapped to independently computed plaintext hashes.
# This preserves verification when Kaggle decompresses files during upload.
KNOWN_DECOMPRESSED_SHA256 = {
    '10f1d0d10888d05a0f52829deaa6a06412dedbf70448caed47e0a0d2708471c0': '72129fad3414f99bc6d34e0c8e0a981ad3d9070417dd23c1a2570021ebddbfac',
    '8a76b0907353550020dac7659d55cf5178aff8d40ef402289e6ee899f8ddf0aa': '61ee30c2d795d28fb0a6c45f92939dbf3324c1c58a9d235e0727e2ef7214f31c',
}

def open_jsonl(path):
    import gzip
    path = pathlib.Path(path)
    if path.exists():
        return gzip.open(path, 'rt', encoding='utf-8')
    plain = path.with_suffix('')
    if path.name.endswith('.jsonl.gz') and plain.is_file():
        return plain.open('rt', encoding='utf-8')
    raise FileNotFoundError(f'Neither {path} nor {plain} exists')

def verify_dataset_file(root, name, info):
    path = pathlib.Path(root) / name
    if path.is_file():
        expected = info['sha256']
    elif name.endswith('.jsonl.gz') and path.with_suffix('').is_file():
        path = path.with_suffix('')
        expected = info.get('uncompressed_sha256') or KNOWN_DECOMPRESSED_SHA256.get(info['sha256'])
        if not expected:
            raise ValueError('No independently recorded plaintext hash for ' + name)
    else:
        raise FileNotFoundError('Missing dataset file: ' + str(path))
    if sha256(path) != expected:
        raise ValueError('Dataset hash mismatch: ' + name)

# Native corpus ingestion. Heavy packages are imported lazily so the config checker stays light.
class LocalCorpus:
    """Disk-backed token corpus with document boundaries and exact byte accounting.

    .bin: little-endian uint32, sidecar <file>.meta.json pins tokenizer_sha256.
    .pt: weights_only tensor/list or {'input_ids': tensor, 'tokenizer_sha256': str}.
    BOS/EOS delimit documents; trailing PAD is accepted only for 2D tensor rows.
    Other special tokens are rejected: their display strings are not original text bytes.
    """
    def __init__(self,tokenizer_path,cache_dir):
        import numpy as np
        from tokenizers import Tokenizer
        self.np=np;self.tokenizer_path=pathlib.Path(tokenizer_path);self.pin=sha256(tokenizer_path)
        self.tokenizer=Tokenizer.from_file(str(tokenizer_path));self.tokenizer.no_padding();self.tokenizer.no_truncation()
        artifact=json.loads(self.tokenizer_path.read_text(encoding='utf8'));self.special={v['id'] for v in artifact['added_tokens'] if v['special']}
        self.bytes,self.lengths=exact_byte_buffers(tokenizer_path,32)
        self.lengths=np.asarray(self.lengths,dtype=np.int64);self.docs=[];self.sources=[];self.offset=0;self.slice_by_offset={};self.labels=["slice_0"];self.current_slice="slice_0"
        cache=pathlib.Path(cache_dir);cache.mkdir(parents=True,exist_ok=True);self.path=cache/'tokens.u32';self.writer=self.path.open('wb')
    def add(self,ids,text=None):
        np=self.np;arr=np.asarray(ids)
        if arr.ndim!=1 or arr.dtype.kind not in 'iu':raise ValueError('Token IDs must be a one-dimensional integer sequence')
        if not len(arr):return
        if int(arr.min())<0 or int(arr.max())>=131072:raise ValueError('Token ID outside pinned vocabulary')
        if any(int(v) in self.special for v in arr):raise ValueError('Non-text special token in scored document')
        raw=b''.join(bytes(self.bytes[int(i)*32:int(i)*32+int(self.lengths[int(i)])]) for i in arr)
        raw.decode('utf8',errors='strict')
        if text is not None and raw!=text.encode('utf8'):raise ValueError('Tokenizer bytes do not match original UTF-8 text')
        self.slice_by_offset[self.offset]=self.current_slice
        arr=arr.astype('<u4',copy=False);arr.tofile(self.writer);self.docs.append((self.offset,len(arr),hashlib.sha256(raw).hexdigest(),len(raw)));self.offset+=len(arr)
    def add_text(self,text):
        if not isinstance(text,str):raise ValueError('JSONL text must be a string')
        if text:self.add(self.tokenizer.encode(text,add_special_tokens=False).ids,text)
    def add_stream(self,ids,trim_padding=False,max_documents=0):
        np=self.np;a=np.asarray(ids)
        if a.ndim!=1 or a.dtype.kind not in 'iu':raise ValueError('Pretokenized stream must contain integers')
        if trim_padding:
            end=len(a)
            while end and int(a[end-1])==130726:end-=1
            a=a[:end]
        start=0
        # EOS/BOS are structural delimiters, not BPB targets. No arbitrary chunk crosses them.
        boundaries=np.flatnonzero((a==36)|(a==130725))
        for end in boundaries:
            if end>start:self.add(a[start:int(end)])
            start=int(end)+1
            if max_documents and len(self.docs)>=max_documents:return
        if start<len(a):self.add(a[start:])
    def load(self,path,max_documents=0):
        import gzip,torch
        path=pathlib.Path(path)
        if path.is_dir():
            candidates=sorted(p for p in path.rglob('*') if p.is_file() and (p.suffix in ('.bin','.pt','.jsonl','.txt') or p.name.endswith('.jsonl.gz')))
            binary=[p for p in candidates if p.suffix in ('.bin','.pt')];paths=binary or candidates
        else:paths=[path]
        if not paths:raise ValueError('Dataset path contains no supported shards')
        for file in paths:
            self.current_slice=self.labels[0]
            self.sources.append({'path':str(file.resolve()),'sha256':sha256(file)})
            if file.suffix in ('.bin','.pt'):
                meta_path=pathlib.Path(str(file)+'.meta.json');meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
                self.current_slice=meta.get('slice_id',self.labels[0])
                if self.current_slice not in self.labels:raise ValueError('Unknown shard slice_id')
                if meta_path.exists():self.sources.append({'path':str(meta_path.resolve()),'sha256':sha256(meta_path)})
                if file.suffix=='.bin':
                    if meta.get('dtype','uint32') not in ('uint32','<u4'):raise ValueError('Only little-endian uint32 .bin is supported')
                    if file.stat().st_size%4:raise ValueError('Truncated uint32 .bin shard')
                    if meta.get('tokenizer_sha256')!=self.pin:raise ValueError('Pretokenized .bin requires matching <file>.meta.json tokenizer_sha256')
                    data=self.np.memmap(file,dtype='<u4',mode='r');self.add_stream(data,max_documents=max_documents)
                else:
                    data=torch.load(file,map_location='cpu',weights_only=True)
                    if isinstance(data,dict):
                        if data.get('tokenizer_sha256',meta.get('tokenizer_sha256'))!=self.pin:raise ValueError('Pretokenized .pt tokenizer fingerprint missing/mismatched')
                        data=data['input_ids']
                    elif meta.get('tokenizer_sha256')!=self.pin:raise ValueError('Tensor .pt requires matching tokenizer sidecar')
                    if not isinstance(data,torch.Tensor) or data.ndim not in (1,2):raise ValueError('.pt requires integer tensor input_ids with 1 or 2 dimensions')
                    if data.dtype not in (torch.int16,torch.int32,torch.int64,torch.uint8):raise ValueError('Floating .pt token tensors are rejected')
                    if data.ndim==1:self.add_stream(data.numpy(),max_documents=max_documents)
                    else:
                        for row in data:
                            self.add_stream(row.numpy(),trim_padding=True,max_documents=max_documents)
                            if max_documents and len(self.docs)>=max_documents:break
            else:
                opener=gzip.open if file.name.endswith('.gz') else open
                with opener(file,'rt',encoding='utf8',newline='') as stream:
                    for line in stream:
                        if file.suffix=='.txt':self.add_text(line)
                        elif line.strip():
                            row=json.loads(line);self.current_slice=row.get('slice_id',row.get('domain',self.labels[0])) if len(self.labels)>1 else self.labels[0]
                            if self.current_slice not in self.labels:raise ValueError('Unknown slice label: '+str(self.current_slice))
                            self.add_text(row['text'])
                        if max_documents and len(self.docs)>=max_documents:break
            if max_documents and len(self.docs)>=max_documents:break
        # Fingerprinting and delimiter scanning inspect the shard; byte reconstruction/cache writing stops at the document limit.
        if max_documents:self.docs=self.docs[:max_documents]
    def finish(self):
        self.writer.close()
        if not self.docs:raise ValueError('No nonempty text documents found')
        self.tokens=self.np.memmap(self.path,dtype='<u4',mode='r')
        return self
    def close(self):
        if not self.writer.closed:self.writer.close()
        if hasattr(self,'tokens'):self.tokens._mmap.close()

class CorpusWindows:
    def __init__(self,corpus,documents,seq_len):
        self.corpus=corpus;self.seq_len=seq_len;self.windows=[]
        for d in documents:
            offset,length,_,_=corpus.docs[d]
            for start in range(0,length,seq_len):self.windows.append((offset,start,min(seq_len,length-start)))
        if not self.windows:raise ValueError('No dataset windows')
    def __len__(self):return len(self.windows)
    def batch(self,indices,device):
        import torch
        np=self.corpus.np;x=np.full((len(indices),self.seq_len),130726,dtype=np.int64);y=np.full_like(x,-100);bytes_=0
        for row,index in enumerate(indices):
            offset,start,n=self.windows[int(index)];ids=self.corpus.tokens[offset+start:offset+start+n].astype(np.int64)
            y[row,:n]=ids;x[row,0]=130725 if start==0 else int(self.corpus.tokens[offset+start-1]);x[row,1:n]=ids[:-1]
            bytes_+=int(self.corpus.lengths[ids].sum())
        return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),bytes_

class DeterministicSampler:
    def __init__(self,size,seed):
        import torch
        self.size=size;self.generator=torch.Generator(device='cpu').manual_seed(seed);self.order=[];self.position=0
    def batch(self,n):
        import torch
        result=[]
        while len(result)<n:
            if self.position==len(self.order):self.order=torch.randperm(self.size,generator=self.generator).tolist();self.position=0
            take=min(n-len(result),len(self.order)-self.position);result.extend(self.order[self.position:self.position+take]);self.position+=take
        return result

def document_split(corpus):
    """Keep identical documents in one partition; independent of model RNG."""
    unique=sorted({d[2] for d in corpus.docs})
    if len(unique)<2:raise ValueError('Automatic validation split needs >=2 distinct documents; provide --validation-data-path')
    heldout=set(unique[:max(1,len(unique)//5)]);train=[];validation=[]
    for i,doc in enumerate(corpus.docs):(validation if doc[2] in heldout else train).append(i)
    return train,validation

def curriculum_at(config,step):
    stages=config['data_curriculum']
    if not stages['enabled']:return 'disabled',dict(seq_len=config['training']['sequence_length'],rope_theta=config['attention']['rope_theta'],moe_aux_loss_coeff=config['moe']['aux_coefficient'])
    position=(step-1)/config['training']['steps'];end=0.
    for name in ['stage_1_warmup','stage_2_main','stage_3_anneal']:
        end+=stages[name]['step_fraction']
        if position<end or name=='stage_3_anneal':return name,stages[name]

class CurriculumEngine:
    """Serializable smooth schedule and excess-loss weighting hook, not a DoReMi reproduction."""
    def __init__(self,config):
        self.config=config;self.step=0;self.tokens_seen=0;self.online_weights=None;self.last_feedback_step=0
    def at(self,step):
        import math
        c=self.config;cur=c['data_curriculum'];name,target=curriculum_at(c,step);state=dict(target);progress=1.
        names=['stage_1_warmup','stage_2_main','stage_3_anneal'];index=names.index(name) if name in names else 0
        if cur['enabled'] and index:
            boundary=math.ceil(sum(cur[n]['step_fraction'] for n in names[:index])*c['training']['steps'])+1
            ramp=cur['transition_ramp_steps'];progress=min(1.,max(0.,(step-boundary)/max(1,ramp))) if ramp else 1.
            alpha=(1-math.cos(math.pi*progress))/2 if cur['transition_curve']=='cosine' else progress
            previous=cur[names[index-1]]
            for key in ['moe_aux_loss_coeff','router_temperature']:
                state[key]=(1-alpha)*previous[key]+alpha*target[key]
            state['mixture_weights']=[(1-alpha)*a+alpha*b for a,b in zip(previous['mixture_weights'],target['mixture_weights'])]
        state.setdefault('router_temperature',1.);state.setdefault('mixture_weights',cur['stage_1_warmup']['mixture_weights'])
        if cur['weighting_mode']=='online_excess_loss' and self.online_weights is not None:state['mixture_weights']=self.online_weights[:]
        floor=cur['min_domain_weight'];n=len(state['mixture_weights']);weights=state['mixture_weights'];total=sum(weights)
        state['mixture_weights']=[floor+(1-n*floor)*w/total for w in weights]
        state.update(stage=name,transition_progress=progress)
        return state
    def observe_excess(self,losses,reference_losses,step):
        import math
        cur=self.config['data_curriculum']
        if cur['weighting_mode']!='online_excess_loss':return
        if len(losses)!=len(cur['domain_labels']) or len(reference_losses)!=len(losses):raise ValueError('Excess-loss slice dimensions differ')
        if not all(math.isfinite(x) for x in losses+reference_losses):raise ValueError('Nonfinite excess-loss feedback')
        weights=self.online_weights or self.at(step)['mixture_weights'];scores=[math.log(max(w,1e-30))+cur['online_weight_lr']*max(0.,loss-ref) for w,loss,ref in zip(weights,losses,reference_losses)]
        peak=max(scores);values=[math.exp(v-peak) for v in scores];total=sum(values);self.online_weights=[v/total for v in values];self.last_feedback_step=step
    def state_dict(self):return dict(step=self.step,tokens_seen=self.tokens_seen,online_weights=self.online_weights,last_feedback_step=self.last_feedback_step)
    def load_state_dict(self,state):
        for key,value in state.items():setattr(self,key,value)

class MixtureSampler:
    """Isolated per-slice epoch permutations; every cursor/generator is checkpointable."""
    def __init__(self,windows,labels,seed):
        import torch
        self.generator=torch.Generator().manual_seed(seed);self.labels=labels;self.groups=[];self.orders=[[] for _ in labels];self.positions=[0 for _ in labels]
        for label in labels:self.groups.append([i for i,(offset,start,n) in enumerate(windows.windows) if windows.corpus.slice_by_offset[offset]==label])
        if any(not group for group in self.groups):raise ValueError('Every configured training slice must contain at least one window')
    def batch(self,n,weights):
        import torch
        result=[]
        for _ in range(n):
            group=int(torch.multinomial(torch.tensor(weights,dtype=torch.float64),1,generator=self.generator)) if len(weights)>1 else 0
            if self.positions[group]==len(self.orders[group]):
                order=torch.randperm(len(self.groups[group]),generator=self.generator).tolist();self.orders[group]=[self.groups[group][j] for j in order];self.positions[group]=0
            result.append(self.orders[group][self.positions[group]]);self.positions[group]+=1
        return result
    def state_dict(self):return dict(generator=self.generator.get_state().clone(),orders=[order[:] for order in self.orders],positions=self.positions[:])
    def load_state_dict(self,state):self.generator.set_state(state['generator']);self.orders=[order[:] for order in state['orders']];self.positions=state['positions'][:]

def token_budget_batch(windows,indices,remaining,device):
    """Truncate only the final microbatch's targets, retaining exact byte accounting."""
    x,y,_=windows.batch(indices,device);n=0;rows=0
    for row in range(len(indices)):
        valid=int(y[row].ne(-100).sum());take=min(valid,remaining-n)
        if take<=0:break
        y[row,take:]=-100;n+=take;rows+=1
    x=x[:rows];y=y[:rows];ids=y[y.ne(-100)].detach().cpu().numpy();bytes_=int(windows.corpus.lengths[ids].sum())
    return x,y,bytes_
