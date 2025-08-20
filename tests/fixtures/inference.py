from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from ocp_resources.serving_runtime import ServingRuntime

from utilities.constants import RuntimeTemplates, KServeDeploymentType
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def vllm_cpu_runtime(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_pod: Pod,
    minio_service: Service,
    minio_data_connection: Secret,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="vllm-runtime-cpu-fp16",
        namespace=model_namespace.name,
        template_name=RuntimeTemplates.VLLM_CUDA,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image="quay.io/rh-aiservices-bu/vllm-cpu-openai-ubi9"
        "@sha256:ada6b3ba98829eb81ae4f89364d9b431c0222671eafb9a04aa16f31628536af2",
        containers={
            "kserve-container": {
                "args": [
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def qwen_isvc_url(qwen_isvc: InferenceService) -> str:
    return f"http://{qwen_isvc.name}-predictor.{qwen_isvc.namespace}.svc.cluster.local:8032/v1"
