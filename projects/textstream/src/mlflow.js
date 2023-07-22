const headers = {
    "Content-Type": "application/json",
}

class MLflow {
    constructor({endpoint}) {
        this.endpoint = endpoint ? endpoint : 'http://localhost:5000'
        this.activeRunId = null
    }

    async createRun(experimentId, name) {
        const body = {
            experiment_id: experimentId,
            run_name: name,
            start_time: Date.now(),
        }
        const response = await fetch(`${this.endpoint}/api/2.0/mlflow/runs/create`, {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        })
        const json = await response.json()
        this.activeRunId = json.run.info.run_id
        return json
    }

    async logMetric(key, value, step) {
        if (!this.activeRunId) {
            throw new Error('No active run')
        }
        const body = {
            run_id: this.activeRunId,
            key: key,
            value: value,
            timestamp: Date.now(),
            step: step
        }

        const response = await fetch(`${this.endpoint}/api/2.0/mlflow/runs/log-metric`, {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        })
        const json = await response.json()
        return json
    }

    async logParam(key, value) {
        if (!this.activeRunId) {
            throw new Error('No active run')
        }
        const body = {
            run_id: this.activeRunId,
            key: key,
            value: value + '',
        }

        const response = await fetch(`${this.endpoint}/api/2.0/mlflow/runs/log-parameter`, {
            method: 'POST',
            headers,
            body: JSON.stringify(body)
        })
        const json = await response.json()
        return json
    }
}

module.exports = MLflow