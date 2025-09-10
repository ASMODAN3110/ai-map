export interface UploadedFile {
  id: string
  file: File
  name: string
  size: number
  type: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress: number
}

export interface AnalysisMethod {
  id: 'pole-dipole' | 'schlumberger'
  name: string
  description: string
  measurements: number
  icon: string
}

export interface AnalysisProgress {
  stage: 'uploading' | 'processing' | 'analyzing' | 'visualizing' | 'completed' | 'error'
  progress: number
  message: string
  estimatedTime?: number
}

export interface AnalysisResult {
  id: string
  method: AnalysisMethod['id']
  timestamp: Date
  status: 'success' | 'error' | 'warning'
  metrics: {
    accuracy: number
    processingTime: number
    dataPoints: number
  }
  visualizations: {
    pseudoSection2D?: string // Base64 image or URL
    model3D?: string
    resistivityMap?: string
    chargeabilityMap?: string
  }
  data: {
    rawData?: any[]
    processedData?: any[]
    modelOutput?: any[]
  }
}

export interface ModelInfo {
  name: string
  type: 'cnn-2d' | 'cnn-3d' | 'hybrid'
  version: string
  accuracy: number
  description: string
  parameters: number
}

export interface ApiResponse<T = any> {
  success: boolean
  data?: T
  error?: string
  message?: string
}
