import { AnalysisResult, ApiResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

class ApiClient {
  private baseUrl: string

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const response = await fetch(`${this.baseUrl}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  // Upload de fichiers
  async uploadFiles(files: File[]): Promise<ApiResponse<{ fileIds: string[] }>> {
    const formData = new FormData()
    files.forEach((file) => {
      formData.append('files', file)
    })

    try {
      const response = await fetch(`${this.baseUrl}/api/upload`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed',
      }
    }
  }

  // Démarrer l'analyse
  async startAnalysis(params: {
    method: string
    model: string
    fileIds: string[]
  }): Promise<ApiResponse<{ analysisId: string }>> {
    return this.request('/api/analysis/start', {
      method: 'POST',
      body: JSON.stringify(params),
    })
  }

  // Obtenir le statut de l'analyse
  async getAnalysisStatus(analysisId: string): Promise<ApiResponse<{
    status: 'pending' | 'processing' | 'completed' | 'error'
    progress: number
    message: string
    result?: AnalysisResult
  }>> {
    return this.request(`/api/analysis/${analysisId}/status`)
  }

  // Obtenir les résultats d'analyse
  async getAnalysisResults(analysisId: string): Promise<ApiResponse<AnalysisResult>> {
    return this.request(`/api/analysis/${analysisId}/results`)
  }

  // Générer des images à partir de CSV
  async generateImagesFromCSV(
    file: File, 
    method: string, 
    samples: number = 3
  ): Promise<ApiResponse<{
    success: boolean
    method: string
    num_samples: number
    visualizations: {
      pseudo_sections: string[]
      chargeability_maps: string[]
      model_3d: string
    }
    metadata: {
      columns_used?: string[]
      data_shape: number[]
      processing_time?: string
      data_type?: string
    }
  }>> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('method', method)
    formData.append('samples', samples.toString())

    try {
      const response = await fetch(`${this.baseUrl}/api/generate-images`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return { success: true, data }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      }
    }
  }

  // Générer des images d'exemple
  async generateSampleImages(
    method: string, 
    samples: number = 3
  ): Promise<ApiResponse<{
    success: boolean
    method: string
    num_samples: number
    visualizations: {
      pseudo_sections: string[]
      chargeability_maps: string[]
      model_3d: string
    }
    metadata: {
      data_type?: string
      data_shape: number[]
      processing_time?: string
    }
  }>> {
    const formData = new FormData()
    formData.append('method', method)
    formData.append('samples', samples.toString())

    try {
      const response = await fetch(`${this.baseUrl}/api/generate-sample-images`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return { success: true, data }
    } catch (error) {
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'Unknown error' 
      }
    }
  }

  // Obtenir les modèles disponibles
  async getAvailableModels(): Promise<ApiResponse<{
    models: Array<{
      id: string
      name: string
      type: 'cnn-2d' | 'cnn-3d' | 'hybrid'
      accuracy: number
      description: string
    }>
  }>> {
    return this.request('/api/models')
  }

  // Tester la connexion
  async testConnection(): Promise<ApiResponse<{ status: string }>> {
    return this.request('/api/health')
  }
}

// Instance singleton
export const apiClient = new ApiClient()

// Hooks pour React Query (optionnel)
export const useApiClient = () => apiClient

// Types pour les WebSockets (pour le suivi en temps réel)
export interface WebSocketMessage {
  type: 'progress' | 'result' | 'error'
  analysisId: string
  data: any
}

export class WebSocketClient {
  private ws: WebSocket | null = null
  private listeners: Map<string, (data: any) => void> = new Map()

  connect(analysisId: string) {
    const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'
    this.ws = new WebSocket(`${wsUrl}/${analysisId}`)

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data)
        const listener = this.listeners.get(message.type)
        if (listener) {
          listener(message.data)
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    this.ws.onclose = () => {
      console.log('WebSocket connection closed')
    }

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error)
    }
  }

  onMessage(type: string, callback: (data: any) => void) {
    this.listeners.set(type, callback)
  }

  disconnect() {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.listeners.clear()
  }
}

export const wsClient = new WebSocketClient()
