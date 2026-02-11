/**
 * DEPA Training Demo UI v2 - Modular Frontend
 * Dynamically discovers and manages any training scenario
 */

class DepaTrainingApp {
    constructor() {
        this.state = {
            scenarios: [],
            currentScenario: null,
            currentScenarioData: null,
            azureLoggedIn: false,
            pipelineConfig: null,
            eventSource: null,
        };
        
        this.init();
    }

    async init() {
        this.bindEvents();
        await this.loadScenarios();
        await this.checkAzureStatus();
        this.renderCommonSettings();
    }

    bindEvents() {
        // Navigation
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', () => this.switchPage(tab.dataset.page));
        });

        // Azure
        document.getElementById('btnAzureLogin').addEventListener('click', () => this.azureLogin());

        // Scenarios
        document.getElementById('scenarioSelect').addEventListener('change', (e) => {
            if (e.target.value) this.selectScenario(e.target.value);
        });
        document.getElementById('btnRefreshScenarios').addEventListener('click', () => this.refreshScenarios());

        // Pre-deployment steps
        document.getElementById('btnPullContainers').addEventListener('click', () => this.pullContainers());
        document.getElementById('btnPreprocess').addEventListener('click', () => this.runPreprocess());
        document.getElementById('btnSaveModel').addEventListener('click', () => this.runSaveModel());
        document.getElementById('btnTestTraining').addEventListener('click', () => this.runTestTraining());

        // Deploy
        document.getElementById('btnDeploy').addEventListener('click', () => this.deploy());
        document.getElementById('btnDownload').addEventListener('click', () => this.downloadModel());
        
        // Cleanup dropdown
        const btnCleanup = document.getElementById('btnCleanup');
        const cleanupDropdown = document.getElementById('cleanupDropdown');
        btnCleanup.addEventListener('click', (e) => {
            e.stopPropagation();
            btnCleanup.closest('.dropdown').classList.toggle('active');
        });
        document.getElementById('btnCleanupContainer').addEventListener('click', () => this.cleanupContainer());
        document.getElementById('btnCleanupResourceGroup').addEventListener('click', () => this.cleanupResourceGroup());
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown')) {
                document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));
            }
        });

        // Config
        document.getElementById('btnReloadConfig').addEventListener('click', () => this.loadPipelineConfig());
        document.getElementById('btnSaveConfig').addEventListener('click', () => this.savePipelineConfig());

        // Logs
        document.getElementById('btnRefreshLogs').addEventListener('click', () => this.refreshLogs());

        // Settings
        document.getElementById('btnSaveSettings').addEventListener('click', () => this.saveSettings());

        // Modal
        document.getElementById('btnCloseModal').addEventListener('click', () => this.closeModal());
        document.getElementById('btnCancelScript').addEventListener('click', () => this.cancelScript());
        document.querySelector('.modal-backdrop').addEventListener('click', () => this.closeModal());

        // Shutdown
        document.getElementById('btnShutdown').addEventListener('click', () => this.shutdown());
    }

    async shutdown() {
        if (!confirm('Shutdown the Demo UI server?')) return;
        
        this.toast('Shutting down server...', 'info');
        try {
            await fetch('/api/shutdown', { method: 'POST' });
        } catch (e) {
            // Expected - server will close connection
        }
        document.body.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#0d0d0f;color:#fafafa;font-family:system-ui;"><div style="text-align:center;"><h1 style="color:#f0a500;">Server Stopped</h1><p style="color:#a1a1aa;">You can close this tab now.</p></div></div>';
    }

    // ============= Navigation =============

    switchPage(page) {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.toggle('active', t.dataset.page === page));
        document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === `page-${page}`));

        if (page === 'config') this.loadPipelineConfig();
        if (page === 'logs') { this.refreshLogs(); this.checkContainerStatus(); }
        if (page === 'settings') this.loadSettings();
    }

    // ============= Scenarios =============

    async loadScenarios() {
        try {
            const res = await fetch('/api/scenarios');
            this.state.scenarios = await res.json();
            this.renderScenarioDropdown();
        } catch (e) {
            console.error('Failed to load scenarios:', e);
            this.toast('Failed to load scenarios', 'error');
        }
    }

    async refreshScenarios() {
        try {
            const res = await fetch('/api/scenarios/refresh');
            this.state.scenarios = await res.json();
            this.renderScenarioDropdown();
            this.toast('Scenarios refreshed', 'success');
        } catch (e) {
            this.toast('Failed to refresh scenarios', 'error');
        }
    }

    renderScenarioDropdown() {
        const select = document.getElementById('scenarioSelect');
        const current = select.value;
        
        select.innerHTML = '<option value="">Select a scenario...</option>' +
            this.state.scenarios.map(s => 
                `<option value="${s.name}" ${s.name === current ? 'selected' : ''}>${this.formatName(s.name)}</option>`
            ).join('');
    }

    async selectScenario(scenarioName) {
        try {
            const res = await fetch('/api/select-scenario', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ scenario: scenarioName })
            });
            const data = await res.json();
            
            if (data.success) {
                this.state.currentScenario = scenarioName;
                this.state.currentScenarioData = data.scenario;
                // Clear the cached pipeline config when switching scenarios!
                this.state.pipelineConfig = null;
                this.renderScenarioInfo(data.scenario);
                this.renderPipeline(data.scenario);
                this.renderScenarioSettings(data.scenario_vars);
                this.updatePreDeploymentButtons(data.scenario);
                this.toast(`Selected: ${this.formatName(scenarioName)}`, 'success');
            }
        } catch (e) {
            console.error('Failed to select scenario:', e);
            this.toast('Failed to select scenario', 'error');
        }
    }

    renderScenarioInfo(scenario) {
        const info = document.getElementById('scenarioInfo');
        const badge = document.getElementById('pipelineScenarioBadge');
        const settingsTag = document.getElementById('settingsScenarioTag');
        
        badge.textContent = this.formatName(scenario.name);
        settingsTag.textContent = this.formatName(scenario.name);
        
        info.innerHTML = `
            <div class="scenario-details">
                <span class="scenario-name-display">${this.formatName(scenario.name)}</span>
                <span class="scenario-steps">${scenario.description || 'Training scenario'}</span>
                <span class="scenario-vars-count">${Object.keys(scenario.scenario_vars || {}).length} scenario variables</span>
            </div>
        `;
    }

    renderPipeline(scenario) {
        const container = document.getElementById('pipelineContainer');
        const scripts = scenario.deployment_scripts || [];
        
        if (scripts.length === 0) {
            container.innerHTML = `
                <div class="pipeline-empty">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                        <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/>
                    </svg>
                    <p>No deployment scripts found for this scenario</p>
                </div>
            `;
            return;
        }
        
        // Filter out post-deploy scripts and optional scripts for main pipeline
        const mainScripts = scripts.filter(s => !s.post_deploy && !s.optional);
        
        container.innerHTML = mainScripts.map((script, idx) => {
            const key = script.filename.replace('.sh', '');
            return `
                <div class="pipeline-step" data-script="${script.filename}" data-key="${key}">
                    <div class="step-number">${idx + 1}</div>
                    <div class="step-content">
                        <div class="step-name">${script.name}</div>
                        <div class="step-description">${script.description}</div>
                    </div>
                    <span class="step-status pending">Pending</span>
                    <button class="btn btn-outline btn-sm step-run-btn">Run</button>
                </div>
            `;
        }).join('');
        
        // Bind run buttons
        container.querySelectorAll('.step-run-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const step = e.target.closest('.pipeline-step');
                this.runScript(step.dataset.script, step.dataset.key);
            });
        });
    }

    // ============= Azure =============

    async checkAzureStatus() {
        try {
            const res = await fetch('/api/azure/check');
            const data = await res.json();
            
            const statusEl = document.getElementById('azureStatus');
            const notLoggedIn = document.getElementById('azureNotLoggedIn');
            const loggedIn = document.getElementById('azureLoggedIn');
            
            if (data.logged_in) {
                this.state.azureLoggedIn = true;
                statusEl.classList.add('logged-in');
                statusEl.classList.remove('not-logged-in');
                statusEl.querySelector('.status-text').textContent = 'Connected';
                
                notLoggedIn.classList.add('hidden');
                loggedIn.classList.remove('hidden');
                
                document.getElementById('subscriptionName').textContent = data.account.name || '-';
                document.getElementById('userName').textContent = data.account.user?.name || '-';
            } else {
                statusEl.classList.remove('logged-in');
                statusEl.classList.add('not-logged-in');
                statusEl.querySelector('.status-text').textContent = 'Not connected';
                
                notLoggedIn.classList.remove('hidden');
                loggedIn.classList.add('hidden');
            }
        } catch (e) {
            console.error('Azure check failed:', e);
        }
    }

    async azureLogin() {
        this.toast('Starting Azure login...', 'info');
        try {
            const res = await fetch('/api/azure/login');
            const data = await res.json();
            
            if (data.requires_device_code) {
                this.toast('Complete login in browser with the device code', 'info');
                this.pollAzureLogin();
            } else if (data.success) {
                await this.checkAzureStatus();
                this.toast('Logged in successfully', 'success');
            }
        } catch (e) {
            this.toast('Login failed', 'error');
        }
    }

    async pollAzureLogin() {
        let attempts = 0;
        const poll = async () => {
            if (++attempts > 60) return this.toast('Login timeout', 'error');
            const res = await fetch('/api/azure/check');
            const data = await res.json();
            if (data.logged_in) {
                await this.checkAzureStatus();
                this.toast('Logged in successfully', 'success');
            } else {
                setTimeout(poll, 2000);
            }
        };
        setTimeout(poll, 5000);
    }

    // ============= Pre-deployment Steps =============

    updatePreDeploymentButtons(scenario) {
        const localScripts = scenario.local_scripts || {};
        const saveModelBtn = document.getElementById('btnSaveModel');
        
        if (localScripts.has_save_model) {
            saveModelBtn.classList.remove('hidden');
        } else {
            saveModelBtn.classList.add('hidden');
        }
    }

    async pullContainers() {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }

        this.openModal('pull-containers.sh');
        
        const terminal = document.getElementById('terminalOutput');
        terminal.textContent = `$ ./ci/pull-containers.sh\n$ ./scenarios/${this.state.currentScenario}/ci/pull-containers.sh\n\n`;

        try {
            const res = await fetch('/api/pull-containers-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                for (const line of decoder.decode(value).split('\n')) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.line) {
                                terminal.textContent += data.line;
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                            if (data.status) {
                                this.updateModalStatus(data.status);
                            }
                        } catch {}
                    }
                }
            }
        } catch (e) {
            this.updateModalStatus('failed');
        }
    }

    async runLocalScript(scriptName, displayName) {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }

        this.openModal(displayName || scriptName);
        
        const terminal = document.getElementById('terminalOutput');
        terminal.textContent = `$ ./${scriptName}\n\n`;

        try {
            const res = await fetch('/api/run-local-script-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: scriptName })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                for (const line of decoder.decode(value).split('\n')) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.line) {
                                terminal.textContent += data.line;
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                            if (data.status) {
                                this.updateModalStatus(data.status);
                            }
                        } catch {}
                    }
                }
            }
        } catch (e) {
            this.updateModalStatus('failed');
        }
    }

    async runPreprocess() {
        await this.runLocalScript('preprocess.sh', 'preprocess.sh');
    }

    async runSaveModel() {
        await this.runLocalScript('save-model.sh', 'save-model.sh');
    }

    async runTestTraining() {
        await this.runLocalScript('train.sh', 'train.sh');
    }

    // ============= Scripts =============

    async runScript(scriptName, scriptKey) {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }
        if (!this.state.azureLoggedIn) {
            return this.toast('Login to Azure first', 'error');
        }

        this.openModal(scriptName);
        this.updateStepStatus(scriptKey, 'running');
        
        const terminal = document.getElementById('terminalOutput');
        terminal.textContent = `$ ./${scriptName}\n\n`;

        try {
            const res = await fetch('/api/run-script-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ script: scriptName })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                for (const line of decoder.decode(value).split('\n')) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.line) {
                                terminal.textContent += data.line;
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                            if (data.status) {
                                this.updateStepStatus(scriptKey, data.status);
                                this.updateModalStatus(data.status);
                            }
                        } catch {}
                    }
                }
            }
        } catch (e) {
            this.updateStepStatus(scriptKey, 'failed');
            this.updateModalStatus('failed');
        }
    }

    updateStepStatus(key, status) {
        const step = document.querySelector(`.pipeline-step[data-key="${key}"]`);
        if (!step) return;
        
        step.classList.remove('pending', 'running', 'completed', 'failed');
        step.classList.add(status);
        
        const badge = step.querySelector('.step-status');
        badge.className = `step-status ${status}`;
        badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }

    // ============= Deploy =============

    async deploy() {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }
        if (!this.state.azureLoggedIn) {
            return this.toast('Login to Azure first', 'error');
        }

        const contractSeqNo = document.getElementById('contractSeqNo').value;
        this.openModal('deploy.sh');
        
        const terminal = document.getElementById('terminalOutput');
        terminal.textContent = `$ ./deploy.sh -c ${contractSeqNo} -p ../../config/pipeline_config.json\n\n`;

        try {
            const res = await fetch('/api/deploy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    contract_seq_no: contractSeqNo,
                    pipeline_config: this.state.pipelineConfig
                })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                for (const line of decoder.decode(value).split('\n')) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.line) {
                                terminal.textContent += data.line;
                                terminal.scrollTop = terminal.scrollHeight;
                            }
                            if (data.status) {
                                this.updateModalStatus(data.status);
                            }
                        } catch {}
                    }
                }
            }
        } catch (e) {
            this.updateModalStatus('failed');
        }
    }

    async downloadModel() {
        if (!this.state.currentScenarioData) {
            return this.toast('Select a scenario first', 'error');
        }
        
        // Find download script
        const scripts = this.state.currentScenarioData.deployment_scripts || [];
        const downloadScript = scripts.find(s => s.post_deploy || s.filename.includes('download'));
        
        if (downloadScript) {
            this.runScript(downloadScript.filename, downloadScript.filename.replace('.sh', ''));
        } else {
            this.toast('Download script not found', 'error');
        }
    }

    async cleanupContainer() {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }
        if (!confirm('Delete the container instance?')) return;

        // Close dropdown
        document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));

        this.toast('Deleting container...', 'info');
        try {
            const res = await fetch('/api/cleanup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type: 'container' }) });
            const data = await res.json();
            this.toast(data.success ? 'Container deleted' : 'Cleanup failed', data.success ? 'success' : 'error');
        } catch {
            this.toast('Cleanup failed', 'error');
        }
    }

    async cleanupResourceGroup() {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }
        const resourceGroup = this.state.currentScenarioData?.common_vars?.AZURE_RESOURCE_GROUP || 'the resource group';
        if (!confirm(`⚠️ WARNING: This will delete the entire resource group "${resourceGroup}" including:\n\n- Container instance\n- Key Vault (will be purged to free the name)\n- Storage account\n- All other resources in the group\n\nThis action cannot be undone. Continue?`)) return;

        // Close dropdown
        document.querySelectorAll('.dropdown').forEach(d => d.classList.remove('active'));

        this.toast('Deleting resource group...', 'info');
        try {
            const res = await fetch('/api/cleanup', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type: 'resource_group' }) });
            const data = await res.json();
            this.toast(data.success ? 'Resource group deleted' : 'Cleanup failed', data.success ? 'success' : 'error');
        } catch {
            this.toast('Cleanup failed', 'error');
        }
    }

    // ============= Config =============

    async loadPipelineConfig() {
        if (!this.state.currentScenario) {
            document.getElementById('configEditor').value = '// Select a scenario to load configuration';
            return;
        }

        try {
            const res = await fetch('/api/pipeline-config');
            const data = await res.json();
            
            if (data.success) {
                this.state.pipelineConfig = data.config;
                document.getElementById('configEditor').value = JSON.stringify(data.config, null, 2);
                document.getElementById('configFilename').textContent = `pipeline_config.json (${this.state.currentScenario})`;
            } else {
                document.getElementById('configEditor').value = `// Error: ${data.error}`;
            }
        } catch {
            document.getElementById('configEditor').value = '// Failed to load configuration';
        }
    }

    async savePipelineConfig() {
        const editor = document.getElementById('configEditor');
        try {
            const config = JSON.parse(editor.value);
            this.state.pipelineConfig = config;
            
            await fetch('/api/pipeline-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ config })
            });
            
            this.toast('Configuration saved', 'success');
        } catch {
            this.toast('Invalid JSON format', 'error');
        }
    }

    // ============= Logs =============

    async refreshLogs() {
        if (!this.state.currentScenario) {
            document.getElementById('logsOutput').textContent = 'Select a scenario first...';
            return;
        }

        document.getElementById('logsOutput').textContent = 'Loading logs...';
        
        try {
            const res = await fetch('/api/logs');
            const data = await res.json();
            document.getElementById('logsOutput').textContent = data.logs || 'No logs available';
        } catch {
            document.getElementById('logsOutput').textContent = 'Failed to fetch logs';
        }
    }

    streamLogs() {
        if (!this.state.currentScenario) {
            return this.toast('Select a scenario first', 'error');
        }

        const output = document.getElementById('logsOutput');
        const btn = document.getElementById('btnStreamLogs');
        
        if (this.state.eventSource) {
            this.state.eventSource.close();
            this.state.eventSource = null;
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg>Stream';
            return;
        }

        output.textContent = 'Streaming logs...\n';
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>Stop';

        this.state.eventSource = new EventSource('/api/logs/stream');
        this.state.eventSource.onmessage = (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.line) {
                    output.textContent += data.line;
                    output.scrollTop = output.scrollHeight;
                }
            } catch {}
        };
        this.state.eventSource.onerror = () => {
            this.state.eventSource.close();
            this.state.eventSource = null;
            btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5,3 19,12 5,21"/></svg>Stream';
        };
    }

    async checkContainerStatus() {
        if (!this.state.currentScenario) return;
        
        try {
            const res = await fetch('/api/container-status');
            const data = await res.json();
            document.getElementById('containerStatusValue').textContent = data.success ? (data.status?.state || 'Unknown') : 'Not found';
        } catch {
            document.getElementById('containerStatusValue').textContent = 'Error';
        }
    }

    // ============= Settings =============

    renderCommonSettings() {
        const commonVars = [
            { key: 'AZURE_LOCATION', label: 'Azure Location', type: 'select', options: ['northeurope', 'westeurope', 'eastus', 'westus', 'centralindia', 'southeastasia'] },
            { key: 'AZURE_RESOURCE_GROUP', label: 'Resource Group', placeholder: 'depa-train-ccr-demo' },
            { key: 'AZURE_KEYVAULT_ENDPOINT', label: 'Key Vault Endpoint', placeholder: 'your-vault.vault.azure.net' },
            { key: 'AZURE_STORAGE_ACCOUNT_NAME', label: 'Storage Account', placeholder: 'yourstorageaccount' },
            { key: 'CONTAINER_REGISTRY', label: 'Container Registry', placeholder: 'registry.azurecr.io' },
            { key: 'CONTRACT_SERVICE_URL', label: 'Contract Service URL', placeholder: 'https://...' },
        ];

        const grid = document.getElementById('commonVarsGrid');
        grid.innerHTML = commonVars.map(v => {
            if (v.type === 'select') {
                return `
                    <div class="form-group">
                        <label class="form-label">${v.label}</label>
                        <select class="form-select common-var" data-key="${v.key}">
                            ${v.options.map(o => `<option value="${o}">${this.formatName(o)}</option>`).join('')}
                        </select>
                    </div>
                `;
            }
            return `
                <div class="form-group">
                    <label class="form-label">${v.label}</label>
                    <input type="text" class="form-input common-var" data-key="${v.key}" placeholder="${v.placeholder || ''}">
                </div>
            `;
        }).join('');
    }

    renderScenarioSettings(vars) {
        const grid = document.getElementById('scenarioVarsGrid');
        
        if (!vars || Object.keys(vars).length === 0) {
            grid.innerHTML = '<p class="empty-hint">No scenario-specific variables</p>';
            return;
        }

        grid.innerHTML = Object.entries(vars).map(([key, value]) => `
            <div class="form-group">
                <label class="form-label">${this.formatVarName(key)}</label>
                <input type="text" class="form-input scenario-var" data-key="${key}" value="${value}">
            </div>
        `).join('');
    }

    loadSettings() {
        fetch('/api/state').then(r => r.json()).then(data => {
            const envVars = data.env_vars || {};
            const scenarioVars = data.scenario_vars || {};

            document.querySelectorAll('.common-var').forEach(el => {
                const key = el.dataset.key;
                if (envVars[key]) {
                    el.value = envVars[key];
                }
            });

            this.renderScenarioSettings(scenarioVars);
        });
    }

    async saveSettings() {
        const envVars = {};
        document.querySelectorAll('.common-var').forEach(el => {
            envVars[el.dataset.key] = el.value;
        });

        const scenarioVars = {};
        document.querySelectorAll('.scenario-var').forEach(el => {
            scenarioVars[el.dataset.key] = el.value;
        });

        try {
            await fetch('/api/update-env', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ env_vars: envVars, scenario_vars: scenarioVars })
            });
            this.toast('Settings saved', 'success');
        } catch {
            this.toast('Failed to save settings', 'error');
        }
    }

    // ============= Modal =============

    openModal(scriptName) {
        document.getElementById('outputModal').classList.add('active');
        document.getElementById('modalScriptName').textContent = scriptName;
        document.getElementById('terminalOutput').textContent = '';
        document.getElementById('modalStatus').className = 'modal-status';
        document.getElementById('modalStatus').innerHTML = '<span class="spinner"></span><span>Running...</span>';
    }

    closeModal() {
        document.getElementById('outputModal').classList.remove('active');
        if (this.state.eventSource) {
            this.state.eventSource.close();
            this.state.eventSource = null;
        }
    }

    updateModalStatus(status) {
        const el = document.getElementById('modalStatus');
        if (status === 'completed') {
            el.className = 'modal-status success';
            el.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22,4 12,14.01 9,11.01"/></svg><span>Completed</span>';
            this.toast('Script completed', 'success');
        } else if (status === 'failed') {
            el.className = 'modal-status error';
            el.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg><span>Failed</span>';
            this.toast('Script failed', 'error');
        }
    }

    async cancelScript() {
        await fetch('/api/cancel', { method: 'POST' });
        this.closeModal();
        this.toast('Script cancelled', 'info');
    }

    // ============= Utilities =============

    formatName(str) {
        return str.replace(/-/g, ' ').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatVarName(key) {
        return key.replace('AZURE_', '').replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    toast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: '<path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><polyline points="22,4 12,14.01 9,11.01"/>',
            error: '<circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>',
            info: '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>'
        };
        
        toast.innerHTML = `
            <svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${icons[type] || icons.info}</svg>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
        `;
        
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = 'slideIn 0.25s ease reverse';
            setTimeout(() => toast.remove(), 250);
        }, 4000);
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.app = new DepaTrainingApp();
});

