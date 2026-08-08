# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the one-card LoRA test suite with the V2 model runner."""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest

# from tests.e2e.pull_request.one_card.lora.test_ilama_lora import test_ilama_lora as test_ilama_lora
from tests.e2e.pull_request.one_card.lora.test_llama32_lora import test_llama_lora as test_llama_lora
from tests.e2e.pull_request.one_card.lora.test_lora_with_spec_decode import (
    test_batch_inference_correctness as test_batch_inference_correctness,
)
# from tests.e2e.pull_request.one_card.lora.test_olmoe_lora import test_olmoe_lora as test_olmoe_lora
from tests.e2e.pull_request.one_card.lora.test_qwen3_multi_loras import (
    test_multi_loras_with_tp_sync as test_multi_loras_with_tp_sync,
)
# from tests.e2e.pull_request.one_card.lora.test_qwen3_reranker_lora import (
#     test_reranker_models_lora as test_reranker_models_lora,
# )


@pytest.fixture(autouse=True)
def model_runner_v2_env() -> Generator[None, None, None]:
    """Run every imported LoRA test with the V2 model runner."""
    with patch.dict(
        os.environ,
        {
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        },
    ):
        yield
