from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import (
    KServeDeploymentType,
    MNT_MODELS,
)
from utilities.inference_utils import create_isvc
from utilities.serving_runtime import ServingRuntimeFromTemplate


GUARDRAILS_ORCHESTRATOR_NAME = "guardrails-orchestrator"


# GuardrailsOrchestrator related fixtures
@pytest.fixture(scope="class")
def guardrails_orchestrator(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    orchestrator_config: ConfigMap,
) -> Generator[GuardrailsOrchestrator, Any, Any]:
    gorch_kwargs = {
        "client": admin_client,
        "name": GUARDRAILS_ORCHESTRATOR_NAME,
        "namespace": model_namespace.name,
        "orchestrator_config": orchestrator_config.name,
        "replicas": 1,
        "wait_for_resource": True,
    }

    if enable_built_in_detectors := request.param.get("enable_built_in_detectors"):
        gorch_kwargs["enable_built_in_detectors"] = enable_built_in_detectors

    if request.param.get("enable_guardrails_gateway"):
        guardrails_gateway_config = request.getfixturevalue(argname="guardrails_gateway_config")
        gorch_kwargs["enable_guardrails_gateway"] = True
        gorch_kwargs["guardrails_gateway_config"] = guardrails_gateway_config.name

    with GuardrailsOrchestrator(**gorch_kwargs) as gorch:
        gorch_deployment = Deployment(name=gorch.name, namespace=gorch.namespace, wait_for_resource=True)
        gorch_deployment.wait_for_replicas()
        yield gorch


@pytest.fixture(scope="class")
def orchestrator_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-nlp",
        namespace=model_namespace.name,
        data=request.param["orchestrator_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_gateway_config(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[ConfigMap, Any, Any]:
    with ConfigMap(
        client=admin_client,
        name="fms-orchestr8-config-gateway",
        namespace=model_namespace.name,
        label={Labels.Openshift.APP: "fmstack-nlp"},
        data=request.param["guardrails_gateway_config_data"],
    ) as cm:
        yield cm


@pytest.fixture(scope="class")
def guardrails_orchestrator_pod(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Pod:
    return list(
        Pod.get(
            namespace=model_namespace.name, label_selector=f"app.kubernetes.io/instance={GUARDRAILS_ORCHESTRATOR_NAME}"
        )
    )[0]


@pytest.fixture(scope="class")
def guardrails_orchestrator_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


@pytest.fixture(scope="class")
def guardrails_orchestrator_health_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    guardrails_orchestrator: GuardrailsOrchestrator,
) -> Generator[Route, Any, Any]:
    yield Route(
        name=f"{guardrails_orchestrator.name}-health",
        namespace=guardrails_orchestrator.namespace,
        wait_for_resource=True,
    )


# ServingRuntimes, InferenceServices, and related resources
# for generation and detection models
@pytest.fixture(scope="class")
def huggingface_sr(
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[ServingRuntime, Any, Any]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="guardrails-detector-runtime-prompt-injection",
        template_name=RuntimeTemplates.GUARDRAILS_DETECTOR_HUGGINGFACE,
        namespace=model_namespace.name,
        containers=[
            {
                "name": "kserve-container",
                "image": "quay.io/trustyai/guardrails-detector-huggingface-runtime:v0.2.0",
                "command": ["uvicorn", "app:app"],
                "args": [
                    "--workers=4",
                    "--host=0.0.0.0",
                    "--port=8000",
                    "--log-config=/common/log_conf.yaml",
                ],
                "env": [
                    {"name": "MODEL_DIR", "value": MNT_MODELS},
                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                ],
                "ports": [{"containerPort": 8000, "protocol": "TCP"}],
            }
        ],
        supported_model_formats=[{"name": "guardrails-detector-huggingface", "autoSelect": True}],
    ) as serving_runtime:
        yield serving_runtime


@pytest.fixture(scope="class")
def prompt_injection_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="prompt-injection-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="deberta-v3-base-prompt-injection-v2",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def prompt_injection_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    prompt_injection_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="prompt-injection-detector-route",
        namespace=model_namespace.name,
        service=prompt_injection_detector_isvc.name,
        wait_for_resource=True,
    )


# Other "helper" fixtures
@pytest.fixture(scope="class")
def openshift_ca_bundle_file(
    admin_client: DynamicClient,
) -> str:
    return create_ca_bundle_file(client=admin_client, ca_type="openshift")


@pytest.fixture(scope="class")
def hap_detector_isvc(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    minio_data_connection: Secret,
    huggingface_sr: ServingRuntime,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="hap-detector",
        namespace=model_namespace.name,
        deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
        model_format="guardrails-detector-huggingface",
        runtime=huggingface_sr.name,
        storage_key=minio_data_connection.name,
        storage_path="granite-guardian-hap-38m",
        wait_for_predictor_pods=False,
        enable_auth=False,
        resources={
            "requests": {"cpu": "1", "memory": "4Gi", "nvidia.com/gpu": "0"},
            "limits": {"cpu": "1", "memory": "4Gi", "nvidia.com/gpu": "0"},
        },
        max_replicas=1,
        min_replicas=1,
        labels={
            "opendatahub.io/dashboard": "true",
        },
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def hap_detector_route(
    admin_client: DynamicClient,
    model_namespace: Namespace,
    hap_detector_isvc: InferenceService,
) -> Generator[Route, Any, Any]:
    yield Route(
        name="hap-detector-route",
        namespace=model_namespace.name,
        service=hap_detector_isvc.name,
        wait_for_resource=True,
    )
