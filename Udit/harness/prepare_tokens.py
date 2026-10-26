"""Build portable uint32 shards with a tokenizer fingerprint; never starts training."""
import argparse,json,pathlib,tempfile
from input_utils import LocalCorpus

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 target=pathlib.Path(a.out)
 if target.exists() or pathlib.Path(str(target)+'.meta.json').exists():raise ValueError('Refusing to overwrite existing shard/metadata')
 target.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory() as tmp:
  corpus=LocalCorpus(a.tokenizer,tmp)
  try:
   corpus.load(a.input);corpus.finish()
   with target.open('wb') as out:
    for offset,length,_,_ in corpus.docs:
     corpus.tokens[offset:offset+length].tofile(out);corpus.np.asarray([36],dtype='<u4').tofile(out)
   meta={'dtype':'uint32','tokenizer_sha256':corpus.pin,'document_delimiter_id':36,'documents':len(corpus.docs),'text_tokens':sum(d[1] for d in corpus.docs),'utf8_bytes':sum(d[3] for d in corpus.docs),'sources':corpus.sources}
   pathlib.Path(str(target)+'.meta.json').write_text(json.dumps(meta,indent=2),encoding='utf8');print(json.dumps(meta,indent=2))
  finally:corpus.close()
if __name__=='__main__':main()
