import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parents[1]
URL_VALIDATOR = (
    SCENARIO_DIR / "deployment" / "azure" / "validate-contract-service-url.sh"
)
REPO_ROOT = SCENARIO_DIR.parents[1]


class ScenarioConfigurationTests(unittest.TestCase):
    def test_union_all_preserves_row_multiplicity(self):
        join_config = json.loads(
            (SCENARIO_DIR / "config" / "join_config.json").read_text()
        )
        joined_dataset = join_config["config"]["joined_dataset"]
        self.assertIn("UNION ALL", joined_dataset["joining_query"])
        self.assertIs(joined_dataset["deduplicate_rows"], False)

        pipeline = json.loads(
            (SCENARIO_DIR / "config" / "pipeline_config.json").read_text()
        )
        self.assertIs(
            pipeline["pipeline"][0]["config"]["joined_dataset"][
                "deduplicate_rows"
            ],
            False,
        )

    def test_generator_example_uses_non_conflicting_name(self):
        generator_config = json.loads(
            (
                REPO_ROOT
                / "build-your-own-scenario"
                / "config"
                / "mule-detection.json"
            ).read_text()
        )
        self.assertEqual(generator_config["scenario_name"], "mule-detection-1")
        self.assertNotEqual(generator_config["scenario_name"], SCENARIO_DIR.name)


class ContractRendererTests(unittest.TestCase):
    def render_environment(self):
        return {
            **os.environ,
            "BANK_1_TDP_DID": "did:web:bank1.example",
            "BANK_2_TDP_DID": "did:web:bank2.example",
            "BANK_3_TDP_DID": "did:web:bank3.example",
            "TDC_DID": "did:web:tdc.example",
            "CCRP_DID": "did:web:ccrp.example",
            "AZURE_KEYVAULT_ENDPOINT": "https://mulevault.vault.azure.net",
            "AZURE_STORAGE_ACCOUNT_NAME": "mulestorage",
            "AZURE_BANK_1_CONTAINER_NAME": "bank1",
            "AZURE_BANK_2_CONTAINER_NAME": "bank2",
            "AZURE_BANK_3_CONTAINER_NAME": "bank3",
            "CONTRACT_ID": "11111111-2222-4333-8444-555555555555",
        }

    def test_contract_renderer_materializes_infrastructure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "contract.json"
            subprocess.run(
                [str(SCENARIO_DIR / "contract" / "prepare-contract.sh"), str(output)],
                env=self.render_environment(),
                check=True,
                capture_output=True,
                text=True,
            )

            contract = json.loads(output.read_text())
            self.assertEqual(
                contract["id"], "11111111-2222-4333-8444-555555555555"
            )
            self.assertEqual(
                contract["tdps"],
                [
                    "did:web:bank1.example",
                    "did:web:bank2.example",
                    "did:web:bank3.example",
                ],
            )
            self.assertEqual(contract["tdc"], "did:web:tdc.example")
            self.assertEqual(contract["ccrp"], "did:web:ccrp.example")
            self.assertEqual(
                [dataset["provider"] for dataset in contract["datasets"]],
                [
                    "did:web:bank1.example",
                    "did:web:bank2.example",
                    "did:web:bank3.example",
                ],
            )
            self.assertEqual(
                [dataset["url"] for dataset in contract["datasets"]],
                [
                    "https://mulestorage.blob.core.windows.net/bank1/data.img",
                    "https://mulestorage.blob.core.windows.net/bank2/data.img",
                    "https://mulestorage.blob.core.windows.net/bank3/data.img",
                ],
            )
            self.assertTrue(
                all(
                    dataset["key"]["properties"]["endpoint"]
                    == "https://mulevault.vault.azure.net"
                    for dataset in contract["datasets"]
                )
            )
            self.assertNotIn("$", output.read_text())

    def test_contract_renderer_rejects_missing_required_value(self):
        env = self.render_environment()
        del env["BANK_2_TDP_DID"]
        result = subprocess.run(
            [str(SCENARIO_DIR / "contract" / "prepare-contract.sh")],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BANK_2_TDP_DID", result.stderr)

    def test_contract_renderer_rejects_non_web_did(self):
        env = self.render_environment()
        env["TDC_DID"] = "did:key:local-test-key"
        result = subprocess.run(
            [str(SCENARIO_DIR / "contract" / "prepare-contract.sh")],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TDC_DID must be a did:web identifier", result.stderr)

    def test_contract_renderer_rejects_duplicate_bank_dids(self):
        env = self.render_environment()
        env["BANK_3_TDP_DID"] = env["BANK_1_TDP_DID"]
        result = subprocess.run(
            [str(SCENARIO_DIR / "contract" / "prepare-contract.sh")],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be distinct", result.stderr)


class DeploymentTrustTests(unittest.TestCase):
    def validate_url(self, url):
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; validate_contract_service_url "$2"',
                "bash",
                str(URL_VALIDATOR),
                url,
            ],
            capture_output=True,
            text=True,
        )

    def test_contract_service_url_validator_rejects_local_endpoints(self):
        local_urls = [
            "http://contracts.example.org",
            "https://localhost:8000",
            "https://localhost.:8000",
            "https://api.localhost",
            "https://127.0.0.2:8000",
            "https://[::1]:8000",
            "https://[0:0:0:0:0:0:0:1]:8000",
            "https://[::ffff:127.0.0.1]:8000",
            "https://0.0.0.0:8000",
            "https://host.docker.internal:8000",
        ]
        for url in local_urls:
            with self.subTest(url=url):
                result = self.validate_url(url)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("HTTPS endpoint", result.stderr)

    def test_contract_service_url_validator_accepts_remote_https(self):
        result = self.validate_url("https://contracts.consortium.example:8443")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_key_import_rejects_loopback_contract_service(self):
        env = {
            **os.environ,
            "AZURE_KEYVAULT_ENDPOINT": "https://mulevault.vault.azure.net",
            "CONTRACT_SERVICE_URL": "https://localhost:8000",
            "REPO_ROOT": str(SCENARIO_DIR.parents[1]),
            "SCENARIO": "mule-detection",
            "TOOLS_HOME": "/tmp/not-used",
        }
        result = subprocess.run(
            [str(SCENARIO_DIR / "deployment" / "azure" / "3-import-keys.sh")],
            cwd=SCENARIO_DIR / "deployment" / "azure",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-loopback HTTPS", result.stderr)

    def test_deploy_rejects_loopback_contract_service(self):
        env = {
            **os.environ,
            "AZURE_KEYVAULT_ENDPOINT": "https://mulevault.vault.azure.net",
            "AZURE_LOCATION": "northeurope",
            "AZURE_RESOURCE_GROUP": "mule-rg",
            "AZURE_STORAGE_ACCOUNT_NAME": "mulestorage",
            "AZURE_BANK_1_CONTAINER_NAME": "bank1",
            "AZURE_BANK_2_CONTAINER_NAME": "bank2",
            "AZURE_BANK_3_CONTAINER_NAME": "bank3",
            "AZURE_OUTPUT_CONTAINER_NAME": "output",
            "CONTRACT_SERVICE_URL": "https://127.0.0.1:8000",
            "CONTAINER_REGISTRY": "example.azurecr.io",
            "TOOLS_HOME": "/tmp/not-used",
        }
        result = subprocess.run(
            [
                str(SCENARIO_DIR / "deployment" / "azure" / "deploy.sh"),
                "-c",
                "2.1",
                "-p",
                "/tmp/not-used.json",
            ],
            cwd=SCENARIO_DIR / "deployment" / "azure",
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("non-loopback HTTPS", result.stderr)


if __name__ == "__main__":
    unittest.main()
