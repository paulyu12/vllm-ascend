# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the two-card LoRA test suite with the V2 model runner."""

import os
from collections.abc import Generator
from unittest.mock import patch

import pytest

from tests.e2e.pull_request.two_card.lora.test_ilama_lora_tp2 import (
    test_ilama_lora_tp2 as test_ilama_lora_tp2,
)
from tests.e2e.pull_request.two_card.lora.test_llama32_lora_tp2 import (
    test_llama_lora_tp2 as test_llama_lora_tp2,
)
from tests.e2e.pull_request.two_card.lora.test_qwen3moe_lora import (
    test_qwen3moe_lora_ep as test_qwen3moe_lora_ep,
)
from tests.e2e.pull_request.two_card.lora.test_qwen3moe_lora import (
    test_qwen3moe_lora_multi_id_ep as test_qwen3moe_lora_multi_id_ep,
)
from tests.e2e.pull_request.two_card.lora.test_qwen3moe_lora import (
    test_qwen3moe_lora_tp as test_qwen3moe_lora_tp,
)
from tests.e2e.pull_request.two_card.lora.test_qwen35_densemodel_lora_tp import (
    test_qwen35_text_lora as test_qwen35_text_lora,
)


@pytest.fixture(autouse=True)
def model_runner_v2_env() -> Generator[None, None, None]:
    """Run every imported LoRA test with the V2 model runner."""
    with patch.dict(os.environ, {"VLLM_USE_V2_MODEL_RUNNER": "1"}):
        yield
