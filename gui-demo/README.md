# Interactive GUI Demo

The Interactive GUI Demo provides a web-based interface to explore and execute DEPA Training scenarios. It automatically discovers all available scenarios in the repository and provides an intuitive interface to configure and run them.

## Features

- **Auto-discovery**: Automatically detects all scenarios in the `scenarios/` directory
- **Dynamic Configuration**: Parses `export-variables.sh` files to extract environment variables for each scenario
- **User-friendly Interface**: Simple web UI for configuring and launching training scenarios

## Quick Start

From the repository root, launch the demo:

```bash
./launch-demo.sh
```

The web interface will be available at:
- **Local**: http://localhost:5001
- **Network**: http://<your-ip>:5001

Press `Ctrl+C` or click the "Shutdown" button in the web interface to stop the server.

## Requirements

The demo requires:
- A signed electronic contract for the scenario you want to execute, and the endpoint of the contract service.
- An Azure subscription with the necessary resources to deploy the scenario. Key Vault admin permissions required to import encryption keys. (Stay tuned for CCR on other cloud platforms.)
- Flask and Flask-CORS (installed automatically via `requirements.txt`)
- All prerequisites from the main repository (installed via `install-prerequisites.sh`)

## Steps to Execute a Scenario

1. Login to your Azure subscription.
2. Select a scenario from the dropdown menu.
3. Configure the environment variables for the scenario in the **Settings** tab.
4. Create cloud resources for the scenario - storage containers and key vault.
5. Import encryption keys from Key Vault and bind them to the CCE Policy.
6. Encrypt the data and model and upload them to the storage containers.
7. Modify the training pipeline (if needed) in the **Configurations** tab.
8. Deploy the Confidential Clean Room (CCR), using the contract sequence number and pipeline configuration.
9. Monitor the progress of the training within the CCR in the **Logs** tab. This can take a lot of time depending on the scenario and the data size.
10. Once training is complete, download the results of the training scenario.
11. Clean up the deployment and cloud resources.

## How It Works

1. **Scenario Discovery**: The demo scans the `scenarios/` directory and identifies all available training scenarios
2. **Variable Parsing**: For each scenario, it reads the `export-variables.sh` file to extract environment variables
3. **Configuration UI**: The web interface presents a form with all discovered variables, allowing you to configure them
4. **Execution**: When you submit the form, the demo executes the scenario with your configuration
5. **Monitoring**: Real-time logs from the CCR are streamed to the web interface so you can monitor progress

## Architecture

The demo consists of:
- **Backend** (`app.py`): Flask server that handles scenario discovery, variable parsing, and execution
- **Frontend** (`templates/`, `static/`): Web interface for scenario selection and configuration
- **Launch Script** (`../launch-demo.sh`): Convenience script to set up and start the demo

## Adding New Scenarios

The demo automatically works with any new scenario you create using the [Build Your Own Scenario](../build-your-own-scenario/README.md) guide.

No code changes to the demo are required when adding new scenarios.

## Troubleshooting

1. Azure login issues: In case Azure login through the web interface fails, try logging in manually using the Azure CLI, and then reload the web interface.
```bash
az login --use-device-code
```
2. Scenario discovery issues: If the scenario dropdown menu is empty, try refreshing the scenarios list.
3. Deployment issues: If the deployment fails, try deploying manually using the Azure CLI.

If you encounter any other issues, please check the **Logs** tab for error messages. If the problem persists, please open an issue in the [GitHub repository](https://github.com/iSPIRT/depa-training/issues).