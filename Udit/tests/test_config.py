"""Adversarial config checks: reject broken inputs rather than only testing success."""
import copy,json,pathlib,sys,tempfile,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'harness'))
from check_config import load_config
from source_audit import audit_text
BASE=ROOT/'configs/config_v0.yaml';TINY=ROOT/'configs/tiny_moe.yaml'
class ConfigTests(unittest.TestCase):
    def test_valid(self):self.assertEqual(load_config(BASE,TINY)[1],[])
    def test_open_production_reports_missing(self):self.assertTrue(any('MISSING_VALUE' in e for e in load_config(BASE)[1]))
    def mutated(self,path,value):
        c=json.loads(json.dumps(__import__('yaml').safe_load(TINY.read_text())));target=c
        for key in path[:-1]:target=target.setdefault(key,{})
        target[path[-1]]=value
        def emit(obj,indent=0):
            lines=[]
            for k,v in obj.items():
                if isinstance(v,dict):lines.append(' '*indent+k+':');lines.extend(emit(v,indent+2))
                else:lines.append(' '*indent+k+': '+json.dumps(v)+' # source: test mutation')
            return lines
        with tempfile.TemporaryDirectory() as tmp:
            p=pathlib.Path(tmp)/'override.yaml';p.write_text('\n'.join(emit(c)))
            return load_config(BASE,p)[1]
    def test_head_geometry(self):self.assertTrue(self.mutated(['attention','qk_head_dim'],31))
    def test_gqa(self):self.assertTrue(self.mutated(['attention','kv_heads'],2))
    def test_topk(self):self.assertTrue(self.mutated(['moe','num_experts_per_tok'],9))
    def test_ke_shape(self):self.assertTrue(self.mutated(['embedding','output_dim'],128))
    def test_precision(self):self.assertTrue(self.mutated(['precision','compute_dtype'],'fp8'))
    def test_eps(self):self.assertTrue(self.mutated(['model','norm_eps'],0))
    def test_padding(self):self.assertTrue(self.mutated(['tokenizer','padded_vocab_size'],131073))
    def test_midpoint_not_silent(self):self.assertTrue(self.mutated(['model','residual'],'midpoint'))
    def test_invalid_zero_stage(self):self.assertTrue(self.mutated(['distributed','zero_stage'],2))
    def test_invalid_curriculum_fraction(self):self.assertTrue(self.mutated(['data_curriculum','stage_1_warmup','step_fraction'],0.5))
    def test_decreasing_context(self):self.assertTrue(self.mutated(['data_curriculum','stage_1_warmup','seq_len'],128))
    def test_decreasing_theta(self):self.assertTrue(self.mutated(['data_curriculum','stage_1_warmup','rope_theta'],10000000))
    def test_increasing_aux(self):self.assertTrue(self.mutated(['data_curriculum','stage_3_anneal','moe_aux_loss_coeff'],1.0))
    def test_unknown_tap(self):self.assertTrue(self.mutated(['explainability','sae_tap_points'],['unknown']))
    def test_invalid_probe_k(self):self.assertTrue(self.mutated(['explainability','sae_probe','top_k'],65))
    def test_overlap_rejected(self):self.assertTrue(self.mutated(['distributed','overlap_comm'],True))
    def test_missing_source(self):self.assertIn('MISSING_SOURCE',audit_text('x: 1')[0]['issues'])
    def test_quoted_source_is_not_comment(self):self.assertIn('MISSING_SOURCE',audit_text('x: "# source: fake"')[0]['issues'])
    def test_false_zero_valid(self):self.assertFalse(any(r['issues'] for r in audit_text('x: false # source: test\ny: 0 # source: test')))
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):audit_text('x: 1 # source: test\nx: 2 # source: test')
    def test_null_rejected(self):self.assertIn('MISSING_VALUE',audit_text('x: null # source: test')[0]['issues'])
if __name__=='__main__':unittest.main()
