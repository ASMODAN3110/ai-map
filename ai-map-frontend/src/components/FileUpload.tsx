import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileText } from 'lucide-react'
import { cn } from '@/lib/utils'

interface FileUploadProps {
  onFileUpload: (files: File[]) => void
  maxFiles?: number
  maxSize?: number
  acceptedTypes?: string[]
}

export function FileUpload({ 
  onFileUpload, 
  maxFiles = 10, 
  maxSize = 10 * 1024 * 1024, // 10MB
  acceptedTypes = ['.dat', '.csv', '.txt', '.xlsx']
}: FileUploadProps) {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    onFileUpload(acceptedFiles)
  }, [onFileUpload])

  const { getRootProps, getInputProps, isDragActive, fileRejections } = useDropzone({
    onDrop,
    maxFiles,
    maxSize,
    accept: {
      'text/plain': ['.txt', '.dat'],
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    }
  })

  return (
    <div className="space-y-3 sm:space-y-4">
      <div
        {...getRootProps()}
        className={cn(
          "upload-zone",
          isDragActive && "dragover"
        )}
      >
        <input {...getInputProps()} />
        <Upload className="w-6 h-6 sm:w-8 sm:h-8 text-muted-foreground mx-auto mb-2" />
        <p className="text-xs sm:text-sm text-muted-foreground mb-2 px-2">
          {isDragActive
            ? "Déposez vos fichiers ici..."
            : "Glissez vos fichiers ici ou cliquez pour sélectionner"}
        </p>
        <div className="space-y-1">
          <p className="text-xs text-muted-foreground">
            Formats: {acceptedTypes.join(', ')}
          </p>
          <p className="text-xs text-muted-foreground">
            Max: {(maxSize / (1024 * 1024)).toFixed(0)}MB
          </p>
        </div>
      </div>

      {fileRejections.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs sm:text-sm text-destructive font-medium">Fichiers rejetés:</p>
          {fileRejections.map(({ file, errors }) => (
            <div key={file.name} className="flex items-center space-x-2 p-2 bg-destructive/10 rounded text-xs sm:text-sm">
              <FileText className="w-3 h-3 sm:w-4 sm:h-4 text-destructive flex-shrink-0" />
              <span className="text-destructive flex-1 truncate min-w-0">{file.name}</span>
              <div className="text-xs text-destructive flex-shrink-0">
                {errors.map(error => error.message).join(', ')}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
