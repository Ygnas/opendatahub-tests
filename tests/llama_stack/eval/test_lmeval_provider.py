import pytest

from tests.llama_stack.constants import LlamaStackProviders
                    "type": "model",
                    "provider_id": LlamaStackProviders.Eval.TRUSTYAI_LMEVAL,
                    "sampling_params": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 10},
                },
                "scoring_params": {},
                "num_examples": 2,
            },
        )

        samples = TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_10MIN,
            sleep=30,
            func=lambda: llama_stack_client.eval.jobs.status(
                job_id=job.job_id, benchmark_id=TRUSTYAI_LMEVAL_ARCEASY
            ).status,
        )

        for sample in samples:
            if sample == "completed":
                break
