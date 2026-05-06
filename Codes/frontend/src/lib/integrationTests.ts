/**
 * API Integration Tests
 * Verify backend and frontend communication
 */

import api from './api/client'

interface TestResult {
    name: string
    status: 'pass' | 'fail'
    message: string
    duration: number
}

const results: TestResult[] = []

async function test(name: string, fn: () => Promise<void>) {
    const start = performance.now()
    try {
        await fn()
        results.push({
            name,
            status: 'pass',
            message: 'OK',
            duration: Math.round(performance.now() - start),
        })
        console.log(`✓ ${name}`)
    } catch (error: any) {
        results.push({
            name,
            status: 'fail',
            message: error.message,
            duration: Math.round(performance.now() - start),
        })
        console.error(`✗ ${name}: ${error.message}`)
    }
}

export async function runIntegrationTests() {
    console.log('🧪 Starting API Integration Tests...\n')

    // Health check
    await test('Backend Health Check', async () => {
        const response = await api.get('/health/')
        if (response.status !== 200) throw new Error('Health check failed')
        if (!response.data.status) throw new Error('No status in response')
    })

    // System status
    await test('System Status Endpoint', async () => {
        const response = await api.get('/system/status/')
        if (!response.data.cpu) throw new Error('Missing CPU info')
        if (!response.data.memory) throw new Error('Missing memory info')
    })

    // GPU info
    await test('GPU Info Endpoint', async () => {
        const response = await api.get('/system/gpu/')
        if (!response.data.type) throw new Error('Missing GPU type')
    })

    // Version info
    await test('Version Endpoint', async () => {
        const response = await api.get('/version/')
        if (!response.data.api_version) throw new Error('Missing API version')
    })

    // List configurations
    await test('List Models Endpoint', async () => {
        const response = await api.get('/config/models/')
        if (!Array.isArray(response.data.models)) throw new Error('Models not an array')
    })

    await test('List Tasks Endpoint', async () => {
        const response = await api.get('/config/tasks/')
        if (!Array.isArray(response.data.tasks)) throw new Error('Tasks not an array')
    })

    // Print summary
    console.log('\n📊 Test Summary:')
    const passed = results.filter((r) => r.status === 'pass').length
    const failed = results.filter((r) => r.status === 'fail').length
    console.log(`   Passed: ${passed}`)
    console.log(`   Failed: ${failed}`)
    console.log(`   Total: ${results.length}`)
    console.log(`   Duration: ${results.reduce((sum, r) => sum + r.duration, 0)}ms\n`)

    if (failed > 0) {
        console.error('❌ Some tests failed!')
        return false
    } else {
        console.log('✅ All tests passed!')
        return true
    }
}

// Export for testing
export default runIntegrationTests
