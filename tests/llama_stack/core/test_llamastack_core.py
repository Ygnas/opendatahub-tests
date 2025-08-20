import pytest

from tests.llama_stack.constants import LlamaStackProviders
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Just respond ACK."},
            ],
            temperature=0,
        )

        assert len(response.choices) > 0, "No response after basic inference on llama-stack server"

        # Check if response has the expected structure and content
        content = response.choices[0].message.content
        assert content is not None, "LLM response content is None"
        assert "ACK" in content, "The LLM didn't provide the expected answer to the prompt"
