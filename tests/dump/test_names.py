from janus.dump.names import canonical


def test_megatron_names():
    assert canonical("embedding") == "embed.out"
    assert canonical("decoder.layers.3.self_attention") == "layers.3.attn.out"
    assert canonical("decoder.layers.0.mlp") == "layers.0.mlp.out"
    assert canonical("decoder.layers.12") == "layers.12.out"
    assert canonical("decoder.final_layernorm") == "final_norm.out"
    assert canonical("output_layer") == "logits"
    assert canonical("decoder.layers.3.mlp.router") is None
    assert canonical("decoder.layers.3.self_attention.linear_qkv") == "layers.3.attn.qkv.out"
    assert canonical("decoder.layers.1.mlp.experts") == "layers.1.mlp.experts.out"
    assert canonical("decoder.layers.1.mlp.experts.linear_fc1") is None
