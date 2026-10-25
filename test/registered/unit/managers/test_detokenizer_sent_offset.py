import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _Tokenizer:
    is_fast = True

    def __init__(self, backend):
        self.backend = backend

    def batch_decode(self, ids, skip_special_tokens, spaces_between_special_tokens):
        return self.backend.decode_batch(ids, skip_special_tokens=skip_special_tokens)


@pytest.mark.parametrize("interval", [1, 10, 25, 50, 75, 150])
def test_incomplete_utf8_never_moves_sent_offset_backwards(interval):
    fixture = json.loads(
        Path(__file__).with_name("detokenizer_offset_fixture.json").read_text()
    )
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={piece: i for i, piece in enumerate(fixture["vocabulary"])}
        )
    )
    tokenizer.decoder = decoders.ByteLevel()
    manager = DetokenizerManager.__new__(DetokenizerManager)
    manager.tokenizer = _Tokenizer(tokenizer)
    manager.decode_status = {}
    manager.vocab_size = tokenizer.get_vocab_size()
    manager.disable_tokenizer_batch_decode = False
    manager.is_tool_call_parser_gpt_oss = False
    prompt, ids = fixture["prompt_ids"], fixture["output_ids"]
    previous = 0
    sent_offset = 0
    chunks = []
    for end in [*range(interval, len(ids), interval), len(ids)]:
        recv = SimpleNamespace(
            rids=["request"],
            decoded_texts=[""],
            decode_ids=[(prompt if previous == 0 else []) + ids[previous:end]],
            read_offsets=[len(prompt)],
            finished_reasons=[None if end < len(ids) else {"type": "length"}],
            no_stop_trim=[False],
            skip_special_tokens=[True],
            spaces_between_special_tokens=[False],
        )
        chunks.extend(manager._decode_batch_token_id_output(recv))
        if end < len(ids):
            current = manager.decode_status["request"].sent_offset
            assert current >= sent_offset
            sent_offset = current
        previous = end
    assert "".join(chunks) == fixture["expected_text"]
    assert not manager.decode_status
