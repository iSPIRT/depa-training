import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCENARIO_DIR = Path(__file__).resolve().parents[1]


class ContractRendererTests(unittest.TestCase):
    def render_environment(self):
        return {
            **os.environ,
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
            self.assertEqual(contract["ccrp"], "did:web:ccrp.example")
            self.assertEqual(contract["tdps"], [])
            self.assertEqual(contract["tdc"], "")
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
        del env["CCRP_DID"]
        result = subprocess.run(
            [str(SCENARIO_DIR / "contract" / "prepare-contract.sh")],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CCRP_DID", result.stderr)


class DeploymentTrustTests(unittest.TestCase):
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
