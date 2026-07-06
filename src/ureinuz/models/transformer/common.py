from ... import nn
from ...configs.transformer import CausalLM

class TransformerCausalLM(CausalLM):
    def __init__(
        self, config, 
        decoder, 
        embedder = lambda c, s: nn.Embedding(c.vocab_size, c.hidden_size, rngs=s), 
        lm_head = lambda c, s: nn.Linear(c.hidden_size, c.vocab_size, bias=False, dtype=c.ldtype, rngs=s), 
        rngs: nn.Rngs = None
    ):
        if rngs is None:
            rngs = nn.Rngs(0)
            
        self.config = config

        self.embed_tokens = embedder(config, rngs)
        self.layers = nn.SequentialStack(
            decoder, config, rngs, 
            num_stack=config.num_hidden_layers
        )
        self.lm_head = lm_head(config, rngs)


    def __call__(self, *args, **kwds):
        return super().__call__(*args, **kwds)

    @classmethod
    def from_pretrained(cls, path_or_repo, config, module_map=None, **kwargs):
        module_map = module_map or []
        if isinstance(module_map, dict):
            module_map = list(module_map.items())
            
        tied = getattr(config, 'tie_word_embeddings', False) or getattr(config, 'use_tie_lm_head', False)
        
        new_module_map = []
        for rule in module_map:
            if len(rule) == 2:
                source, target = rule
                if tied and target == "embed_tokens.embedding":
                    new_module_map.append((source, ["embed_tokens.embedding", "lm_head.weight"], lambda x: [x, x.T]))
                    continue
            new_module_map.append(rule)
            
        return super().from_pretrained(path_or_repo, config=config, module_map=new_module_map, **kwargs)