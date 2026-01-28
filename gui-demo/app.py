#!/usr/bin/env python3
"""
DEPA Training Demo UI v2 - Modular Flask Backend
A dynamic web interface that automatically discovers and runs any DEPA training scenario.

Key Features:
- Auto-discovers scenarios from /scenarios directory
- Parses export-variables.sh to extract all environment variables
- Dynamically detects deployment scripts
- Works with any new scenario without code changes
"""

import os
import re
import json
import subprocess
import threading
import queue
from pathlib import Path
from flask import Flask, render_template, jsonify, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get repository root
REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, cwd=os.path.dirname(__file__)
).stdout.strip() or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCENARIOS_DIR = Path(REPO_ROOT) / "scenarios"


def parse_export_variables(scenario_path: Path) -> dict:
    """
    Parse the export-variables.sh file to extract all environment variables.
    Returns a dict with 'common_vars' and 'scenario_vars' keys.
    """
    export_file = scenario_path / "export-variables.sh"
    
    result = {
        "common_vars": {},
        "scenario_vars": {},
        "all_vars": {},
    }
    
    if not export_file.exists():
        return result
    
    # Common variable names that apply to all scenarios
    common_var_names = {
        "SCENARIO", "REPO_ROOT", "CONTAINER_REGISTRY", "AZURE_LOCATION",
        "AZURE_SUBSCRIPTION_ID", "AZURE_RESOURCE_GROUP", "AZURE_KEYVAULT_ENDPOINT",
        "AZURE_STORAGE_ACCOUNT_NAME", "CONTRACT_SERVICE_URL", "TOOLS_HOME"
    }
    
    with open(export_file, 'r') as f:
        content = f.read()
    
    # Match both `declare -x VAR=value` and `export VAR=value` patterns
    # Also handle values with or without quotes
    patterns = [
        r'declare\s+-x\s+(\w+)=(["\']?)([^"\'\n]*)\2',  # declare -x VAR="value" or VAR=value
        r'export\s+(\w+)=(["\']?)([^"\'\n]*)\2',        # export VAR="value" or VAR=value
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            var_name = match[0]
            var_value = match[2] if len(match) > 2 else match[1]
            
            # Skip dynamic values like $(git rev-parse...) and $REPO_ROOT references
            if var_value.startswith('$(') or var_value.startswith('$'):
                continue
            
            result["all_vars"][var_name] = var_value
            
            if var_name in common_var_names:
                result["common_vars"][var_name] = var_value
            elif var_name.startswith("AZURE_") and "CONTAINER" in var_name:
                result["scenario_vars"][var_name] = var_value
    
    return result


def discover_deployment_scripts(scenario_path: Path) -> list:
    """
    Discover all deployment scripts in the scenario's azure deployment folder.
    Returns a sorted list of script info dicts.
    """
    azure_dir = scenario_path / "deployment" / "azure"
    
    if not azure_dir.exists():
        return []
    
    scripts = []
    
    # Define script metadata
    script_info = {
        "0-create-acr.sh": {"name": "Create Container Registry", "description": "Create Azure Container Registry (optional)", "optional": True},
        "1-create-storage-containers.sh": {"name": "Create Storage Containers", "description": "Set up blob storage containers for data and models"},
        "2-create-akv.sh": {"name": "Create Azure Key Vault", "description": "Create HSM-backed Key Vault for key management and policy enforcement"},
        "3-import-keys.sh": {"name": "Import Keys", "description": "Import your keys and bind them to a Confidential Computing policy, so it's released only to verified, attested environments."},
        "4-encrypt-data.sh": {"name": "Encrypt Data", "description": "Encrypt training datasets and models with imported keys"},
        "5-upload-encrypted-data.sh": {"name": "Upload Encrypted Data", "description": "Upload encrypted data and models to blob storage"},
        "6-download-decrypt-model.sh": {"name": "Download & Decrypt Model", "description": "Download and decrypt the trained model", "post_deploy": True},
    }
    
    for script_file in sorted(azure_dir.glob("*.sh")):
        name = script_file.name
        if name in script_info:
            scripts.append({
                "filename": name,
                "path": str(script_file),
                **script_info[name]
            })
        elif name == "deploy.sh":
            # Skip deploy.sh as it's handled separately
            continue
        elif name not in ["generatefs.sh"]:
            # Include any other numbered scripts
            match = re.match(r'^(\d+)-(.+)\.sh$', name)
            if match:
                num, rest = match.groups()
                scripts.append({
                    "filename": name,
                    "path": str(script_file),
                    "name": rest.replace("-", " ").title(),
                    "description": f"Step {num}: {rest.replace('-', ' ').title()}",
                })
    
    return sorted(scripts, key=lambda x: x["filename"])


def get_pipeline_config(scenario_path: Path) -> dict:
    """Load the pipeline configuration for a scenario."""
    config_path = scenario_path / "config" / "pipeline_config.json"
    
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        return json.load(f)


def discover_scenarios() -> list:
    """
    Discover all scenarios in the scenarios directory.
    Returns a list of scenario info dicts.
    """
    scenarios = []
    
    if not SCENARIOS_DIR.exists():
        return scenarios
    
    for item in sorted(SCENARIOS_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        
        # Check if it has the required structure
        has_azure_deployment = (item / "deployment" / "azure").exists()
        has_export_vars = (item / "export-variables.sh").exists()
        
        if not has_azure_deployment:
            continue
        
        # Parse variables
        vars_info = parse_export_variables(item)
        
        # Get pipeline config
        pipeline_config = get_pipeline_config(item)
        pipeline_steps = []
        if pipeline_config:
            pipeline_steps = [p.get("name", "Unknown") for p in pipeline_config.get("pipeline", [])]
        
        # Get deployment scripts
        deployment_scripts = discover_deployment_scripts(item)
        
        # Check for local scripts
        local_dir = item / "deployment" / "local"
        has_preprocess = (local_dir / "preprocess.sh").exists()
        has_save_model = (local_dir / "save-model.sh").exists()
        has_train = (local_dir / "train.sh").exists()
        
        scenarios.append({
            "name": item.name,
            "path": str(item),
            "has_azure_deployment": has_azure_deployment,
            "has_export_vars": has_export_vars,
            "pipeline_steps": pipeline_steps,
            "description": f"{len(pipeline_steps)} pipeline step(s): {', '.join(pipeline_steps)}" if pipeline_steps else "Training scenario",
            "common_vars": vars_info["common_vars"],
            "scenario_vars": vars_info["scenario_vars"],
            "deployment_scripts": deployment_scripts,
            "local_scripts": {
                "has_preprocess": has_preprocess,
                "has_save_model": has_save_model,
                "has_train": has_train,
            },
        })
    
    return scenarios


# Global state
state = {
    "prerequisites_installed": False,
    "azure_logged_in": False,
    "azure_subscription": None,
    "azure_account_info": None,
    "current_scenario": None,
    "scenarios_cache": None,
    "env_vars": {
        "REPO_ROOT": REPO_ROOT,
        "TOOLS_HOME": f"{REPO_ROOT}/external/confidential-sidecar-containers/tools",
        "AZURE_LOCATION": "northeurope",
        "AZURE_SUBSCRIPTION_ID": "",
        "AZURE_RESOURCE_GROUP": "",
        "AZURE_KEYVAULT_ENDPOINT": "",
        "AZURE_STORAGE_ACCOUNT_NAME": "",
        "CONTAINER_REGISTRY": "",
        "CONTRACT_SERVICE_URL": "",
    },
    "scenario_vars": {},
    "script_status": {},
    "running_process": None,
}


def get_full_env():
    """Get full environment variables including scenario-specific ones."""
    env = os.environ.copy()
    env.update(state["env_vars"])
    env.update(state["scenario_vars"])
    if state["current_scenario"]:
        env["SCENARIO"] = state["current_scenario"]
    return env


def run_command(cmd, cwd=None, stream=False):
    """Run a shell command and return output."""
    env = get_full_env()
    if cwd is None:
        cwd = REPO_ROOT
    
    if stream:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=cwd, env=env, text=True, bufsize=1
        )
        return process
    else:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd, env=env
        )
        return result


# ===================== API Routes =====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def get_state():
    """Get current application state."""
    return jsonify({
        "prerequisites_installed": state["prerequisites_installed"],
        "azure_logged_in": state["azure_logged_in"],
        "azure_subscription": state["azure_subscription"],
        "azure_account_info": state["azure_account_info"],
        "current_scenario": state["current_scenario"],
        "env_vars": state["env_vars"],
        "scenario_vars": state["scenario_vars"],
        "script_status": state["script_status"],
    })


@app.route("/api/scenarios")
def get_scenarios():
    """Get all available scenarios with their configurations."""
    # Cache scenarios for performance
    if state["scenarios_cache"] is None:
        state["scenarios_cache"] = discover_scenarios()
    return jsonify(state["scenarios_cache"])


@app.route("/api/scenarios/refresh")
def refresh_scenarios():
    """Force refresh the scenarios cache."""
    state["scenarios_cache"] = discover_scenarios()
    return jsonify(state["scenarios_cache"])


@app.route("/api/select-scenario", methods=["POST"])
def select_scenario():
    """Select a scenario and load its variables."""
    data = request.json
    scenario_name = data.get("scenario")
    
    # Find the scenario in cache
    scenarios = state["scenarios_cache"] or discover_scenarios()
    scenario = next((s for s in scenarios if s["name"] == scenario_name), None)
    
    if not scenario:
        return jsonify({"success": False, "error": f"Scenario not found: {scenario_name}"})
    
    state["current_scenario"] = scenario_name
    
    # Load scenario-specific variables
    state["scenario_vars"] = scenario.get("scenario_vars", {}).copy()
    
    # Update common vars with scenario defaults (if not already set)
    common_vars = scenario.get("common_vars", {})
    for key, value in common_vars.items():
        if key not in state["env_vars"] or not state["env_vars"][key]:
            state["env_vars"][key] = value
    
    # Reset script status for this scenario
    state["script_status"] = {}
    for script in scenario.get("deployment_scripts", []):
        script_key = script["filename"].replace(".sh", "")
        state["script_status"][script_key] = "pending"
    state["script_status"]["deploy"] = "pending"
    
    return jsonify({
        "success": True,
        "scenario": scenario,
        "scenario_vars": state["scenario_vars"],
        "common_vars": common_vars,
    })


@app.route("/api/update-env", methods=["POST"])
def update_env():
    """Update environment variables."""
    data = request.json
    env_vars = data.get("env_vars", {})
    scenario_vars = data.get("scenario_vars", {})
    
    state["env_vars"].update(env_vars)
    state["scenario_vars"].update(scenario_vars)
    
    return jsonify({"success": True})


@app.route("/api/azure/check")
def check_azure():
    """Check Azure CLI login status."""
    result = run_command("az account show --output json")
    if result.returncode == 0:
        try:
            account_info = json.loads(result.stdout)
            state["azure_logged_in"] = True
            state["azure_subscription"] = account_info.get("id")
            state["azure_account_info"] = account_info
            state["env_vars"]["AZURE_SUBSCRIPTION_ID"] = account_info.get("id", "")
            return jsonify({
                "logged_in": True,
                "account": account_info
            })
        except:
            pass
    
    state["azure_logged_in"] = False
    return jsonify({"logged_in": False})


@app.route("/api/azure/login")
def azure_login():
    """Initiate Azure login via device code flow."""
    result = run_command("az login --use-device-code 2>&1")
    if "To sign in" in result.stdout:
        return jsonify({
            "success": True,
            "message": result.stdout,
            "requires_device_code": True
        })
    return jsonify({
        "success": result.returncode == 0,
        "output": result.stdout + result.stderr
    })


@app.route("/api/azure/subscriptions")
def list_subscriptions():
    """List Azure subscriptions."""
    result = run_command("az account list --output json")
    if result.returncode == 0:
        try:
            return jsonify({"success": True, "subscriptions": json.loads(result.stdout)})
        except:
            pass
    return jsonify({"success": False, "error": result.stderr})


@app.route("/api/azure/set-subscription", methods=["POST"])
def set_subscription():
    """Set the active Azure subscription."""
    data = request.json
    subscription_id = data.get("subscription_id")
    
    result = run_command(f"az account set --subscription {subscription_id}")
    if result.returncode == 0:
        state["azure_subscription"] = subscription_id
        state["env_vars"]["AZURE_SUBSCRIPTION_ID"] = subscription_id
        return jsonify({"success": True})
    return jsonify({"success": False, "error": result.stderr})


@app.route("/api/run-script", methods=["POST"])
def run_script():
    """Run a deployment script."""
    data = request.json
    script_name = data.get("script")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    script_path = SCENARIOS_DIR / scenario / "deployment" / "azure" / script_name
    
    if not script_path.exists():
        return jsonify({"success": False, "error": f"Script not found: {script_path}"})
    
    script_key = script_name.replace(".sh", "")
    state["script_status"][script_key] = "running"
    
    def run_in_background():
        cwd = script_path.parent
        cmd = f"bash {script_path}"
        result = run_command(cmd, cwd=str(cwd))
        
        if result.returncode == 0:
            state["script_status"][script_key] = "completed"
        else:
            state["script_status"][script_key] = "failed"
    
    thread = threading.Thread(target=run_in_background)
    thread.start()
    
    return jsonify({"success": True, "message": f"Started {script_name}"})


@app.route("/api/run-script-stream", methods=["POST"])
def run_script_stream():
    """Run a deployment script with streaming output."""
    data = request.json
    script_name = data.get("script")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    script_path = SCENARIOS_DIR / scenario / "deployment" / "azure" / script_name
    
    if not script_path.exists():
        return jsonify({"success": False, "error": f"Script not found: {script_path}"})
    
    script_key = script_name.replace(".sh", "")
    state["script_status"][script_key] = "running"
    
    def generate():
        cwd = script_path.parent
        cmd = f"bash {script_path}"
        process = run_command(cmd, cwd=str(cwd), stream=True)
        state["running_process"] = process
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            process.wait()
            if process.returncode == 0:
                state["script_status"][script_key] = "completed"
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            else:
                state["script_status"][script_key] = "failed"
                yield f"data: {json.dumps({'status': 'failed'})}\n\n"
        finally:
            state["running_process"] = None
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/run-local-script-stream", methods=["POST"])
def run_local_script_stream():
    """Run a local deployment script with streaming output."""
    data = request.json
    script_name = data.get("script")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    script_path = SCENARIOS_DIR / scenario / "deployment" / "local" / script_name
    
    if not script_path.exists():
        return jsonify({"success": False, "error": f"Script not found: {script_path}"})
    
    script_key = script_name.replace(".sh", "").replace("-", "_")
    state["script_status"][script_key] = "running"
    
    def generate():
        cwd = script_path.parent
        cmd = f"bash {script_path}"
        process = run_command(cmd, cwd=str(cwd), stream=True)
        state["running_process"] = process
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            process.wait()
            if process.returncode == 0:
                state["script_status"][script_key] = "completed"
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            else:
                state["script_status"][script_key] = "failed"
                yield f"data: {json.dumps({'status': 'failed'})}\n\n"
        finally:
            state["running_process"] = None
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/pull-containers-stream", methods=["POST"])
def pull_containers_stream():
    """Run both pull-containers.sh scripts (repo-level and scenario-level) with streaming output."""
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    repo_pull_script = Path(REPO_ROOT) / "ci" / "pull-containers.sh"
    scenario_pull_script = SCENARIOS_DIR / scenario / "ci" / "pull-containers.sh"
    
    if not repo_pull_script.exists():
        return jsonify({"success": False, "error": f"Repo pull-containers.sh not found: {repo_pull_script}"})
    
    if not scenario_pull_script.exists():
        return jsonify({"success": False, "error": f"Scenario pull-containers.sh not found: {scenario_pull_script}"})
    
    state["script_status"]["pull_containers"] = "running"
    
    def generate():
        # First run repo-level pull-containers.sh
        yield f"data: {json.dumps({'line': '=== Pulling common containers ===\n'})}\n\n"
        cwd = repo_pull_script.parent
        cmd = f"bash {repo_pull_script}"
        process = run_command(cmd, cwd=str(cwd), stream=True)
        state["running_process"] = process
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            process.wait()
            if process.returncode != 0:
                state["script_status"]["pull_containers"] = "failed"
                yield f"data: {json.dumps({'status': 'failed', 'line': 'Failed to pull common containers\n'})}\n\n"
                return
        finally:
            state["running_process"] = None
        
        # Then run scenario-specific pull-containers.sh
        yield f"data: {json.dumps({'line': '\n=== Pulling scenario-specific containers ===\n'})}\n\n"
        cwd = scenario_pull_script.parent
        cmd = f"bash {scenario_pull_script}"
        process = run_command(cmd, cwd=str(cwd), stream=True)
        state["running_process"] = process
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            process.wait()
            if process.returncode == 0:
                state["script_status"]["pull_containers"] = "completed"
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            else:
                state["script_status"]["pull_containers"] = "failed"
                yield f"data: {json.dumps({'status': 'failed'})}\n\n"
        finally:
            state["running_process"] = None
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/check-local-script", methods=["POST"])
def check_local_script():
    """Check if a local script exists."""
    data = request.json
    script_name = data.get("script")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    script_path = SCENARIOS_DIR / scenario / "deployment" / "local" / script_name
    exists = script_path.exists()
    
    return jsonify({"success": True, "exists": exists})


@app.route("/api/deploy", methods=["POST"])
def deploy():
    """Run the deploy.sh script with contract number and pipeline config."""
    data = request.json
    contract_seq_no = data.get("contract_seq_no", "1")
    pipeline_config = data.get("pipeline_config")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    deploy_path = SCENARIOS_DIR / scenario / "deployment" / "azure" / "deploy.sh"
    config_path = SCENARIOS_DIR / scenario / "config" / "pipeline_config.json"
    
    if not deploy_path.exists():
        return jsonify({"success": False, "error": "deploy.sh not found"})
    
    # If custom pipeline config provided, save it to a temp file
    if pipeline_config:
        temp_config_path = Path("/tmp") / f"pipeline_config_{scenario}.json"
        with open(temp_config_path, "w") as f:
            json.dump(pipeline_config, f, indent=2)
        config_arg = str(temp_config_path)
    else:
        config_arg = str(config_path)
    
    state["script_status"]["deploy"] = "running"
    
    def generate():
        cwd = deploy_path.parent
        cmd = f"bash {deploy_path} -c {contract_seq_no} -p {config_arg}"
        env = get_full_env()
        env["CONTRACT_SEQ_NO"] = str(contract_seq_no)
        
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(cwd), env=env, text=True, bufsize=1
        )
        state["running_process"] = process
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
            
            process.wait()
            if process.returncode == 0:
                state["script_status"]["deploy"] = "completed"
                yield f"data: {json.dumps({'status': 'completed'})}\n\n"
            else:
                state["script_status"]["deploy"] = "failed"
                yield f"data: {json.dumps({'status': 'failed'})}\n\n"
        finally:
            state["running_process"] = None
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/logs")
def get_logs():
    """Get container logs."""
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    resource_group = state["env_vars"].get("AZURE_RESOURCE_GROUP", "")
    container_name = f"depa-training-{scenario}"
    
    cmd = f"az container logs --name {container_name} --resource-group {resource_group} --container-name depa-training"
    result = run_command(cmd)
    
    return jsonify({
        "success": result.returncode == 0,
        "logs": result.stdout if result.returncode == 0 else result.stderr
    })


@app.route("/api/logs/stream")
def stream_logs():
    """Stream container logs."""
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    resource_group = state["env_vars"].get("AZURE_RESOURCE_GROUP", "")
    container_name = f"depa-training-{scenario}"
    
    def generate():
        cmd = f"az container logs --name {container_name} --resource-group {resource_group} --container-name depa-training --follow"
        process = run_command(cmd, stream=True)
        
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    yield f"data: {json.dumps({'line': line})}\n\n"
        except GeneratorExit:
            process.terminate()
    
    return Response(generate(), mimetype='text/event-stream')


@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    """Delete the ACI container instance or the entire resource group."""
    data = request.json or {}
    cleanup_type = data.get("type", "container")  # Default to container for backward compatibility
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    resource_group = state["env_vars"].get("AZURE_RESOURCE_GROUP", "")
    if not resource_group:
        return jsonify({"success": False, "error": "Resource group not configured"})
    
    if cleanup_type == "resource_group":
        # Delete the entire resource group (this will delete all resources including container, key vault, storage, etc.)
        cmd = f"az group delete --name {resource_group} --yes --no-wait"
        result = run_command(cmd)
        
        # Purge the key vault to free its name (key vaults have a soft delete period)
        # This needs to happen after deletion, but we'll try it anyway - if it fails, that's ok
        keyvault_endpoint = state["env_vars"].get("AZURE_KEYVAULT_ENDPOINT", "")
        if keyvault_endpoint:
            # Extract vault name from endpoint (e.g., "myvault.vault.azure.net" -> "myvault")
            vault_name = keyvault_endpoint.split(".")[0]
            # Try to purge - this may fail if vault doesn't exist or isn't deleted yet, that's ok
            # We'll run it in a separate thread to not block the response
            def purge_vault():
                import time
                time.sleep(2)  # Wait a bit for deletion to start
                purge_result = run_command(f"az keyvault purge --name {vault_name}")
                # Ignore errors - vault might not be deleted yet or might not exist
            
            threading.Thread(target=purge_vault, daemon=True).start()
        
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr,
            "message": f"Resource group {resource_group} deletion initiated"
        })
    else:
        # Delete only the container instance (original behavior)
        container_name = f"depa-training-{scenario}"
        cmd = f"az container delete --name {container_name} --resource-group {resource_group} --yes"
        result = run_command(cmd)
        
        return jsonify({
            "success": result.returncode == 0,
            "output": result.stdout if result.returncode == 0 else result.stderr
        })


@app.route("/api/pipeline-config")
def get_pipeline_config_api():
    """Get the pipeline configuration for the current scenario."""
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    config_path = SCENARIOS_DIR / scenario / "config" / "pipeline_config.json"
    
    if not config_path.exists():
        return jsonify({"success": False, "error": "Pipeline config not found"})
    
    with open(config_path) as f:
        config = json.load(f)
    
    return jsonify({"success": True, "config": config})


@app.route("/api/pipeline-config", methods=["POST"])
def save_pipeline_config():
    """Save the pipeline configuration."""
    data = request.json
    config = data.get("config")
    
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    temp_config_path = Path("/tmp") / f"pipeline_config_{scenario}.json"
    with open(temp_config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    return jsonify({"success": True, "path": str(temp_config_path)})


@app.route("/api/cancel", methods=["POST"])
def cancel_running():
    """Cancel any running process."""
    if state["running_process"]:
        try:
            state["running_process"].terminate()
            state["running_process"] = None
            return jsonify({"success": True})
        except:
            pass
    return jsonify({"success": False, "error": "No process running"})


@app.route("/api/container-status")
def container_status():
    """Get the status of the ACI container."""
    scenario = state["current_scenario"]
    if not scenario:
        return jsonify({"success": False, "error": "No scenario selected"})
    
    resource_group = state["env_vars"].get("AZURE_RESOURCE_GROUP", "")
    container_name = f"depa-training-{scenario}"
    
    cmd = f"az container show --name {container_name} --resource-group {resource_group} --query '{{state: instanceView.state, events: containers[0].instanceView.events}}' --output json"
    result = run_command(cmd)
    
    if result.returncode == 0:
        try:
            status = json.loads(result.stdout)
            return jsonify({"success": True, "status": status})
        except:
            pass
    
    return jsonify({"success": False, "error": result.stderr or "Container not found"})


@app.route("/api/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shutdown the server."""
    import os
    import signal
    
    def stop_server():
        os.kill(os.getpid(), signal.SIGTERM)
    
    # Schedule shutdown after response is sent
    from threading import Timer
    Timer(0.5, stop_server).start()
    
    return jsonify({"success": True, "message": "Server shutting down..."})


@app.route("/api/scenario-info/<scenario_name>")
def get_scenario_info(scenario_name):
    """Get detailed info about a specific scenario."""
    scenario_path = SCENARIOS_DIR / scenario_name
    
    if not scenario_path.exists():
        return jsonify({"success": False, "error": "Scenario not found"})
    
    vars_info = parse_export_variables(scenario_path)
    deployment_scripts = discover_deployment_scripts(scenario_path)
    pipeline_config = get_pipeline_config(scenario_path)
    
    # Check for additional config files
    config_files = []
    config_dir = scenario_path / "config"
    if config_dir.exists():
        for f in config_dir.glob("*.json"):
            config_files.append(f.name)
    
    return jsonify({
        "success": True,
        "name": scenario_name,
        "path": str(scenario_path),
        "common_vars": vars_info["common_vars"],
        "scenario_vars": vars_info["scenario_vars"],
        "all_vars": vars_info["all_vars"],
        "deployment_scripts": deployment_scripts,
        "pipeline_config": pipeline_config,
        "config_files": config_files,
    })


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    # print(f"\n🚀 DEPA Training Demo UI v2 (Modular) starting on http://localhost:{port}\n")
    # print(f"📁 Scenarios directory: {SCENARIOS_DIR}\n")
    # print("Discovered scenarios:")
    # for s in discover_scenarios():
    #     print(f"  - {s['name']}: {s['description']}")
    # print()
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

